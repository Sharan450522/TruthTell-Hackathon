import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from text_aggregator import chunk_text, prepare_content_pieces


logger = logging.getLogger(__name__)


VALID_REVIEW_LABELS = {"fake", "real", "misleading", "satire", "uncertain"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class SQLiteFeedbackStore:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join("data", "truth_tell.db")
        self._memory_connection = None
        if self.db_path == ":memory:":
            self._memory_connection = self._new_connection(self.db_path)
        else:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        self.initialize()

    def _new_connection(self, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def connect(self):
        if self._memory_connection is not None:
            return self._memory_connection
        return self._new_connection(self.db_path)

    def initialize(self):
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    url TEXT,
                    file_name TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS extracted_content (
                    content_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp_sec REAL,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES sources(source_id)
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    final_status TEXT NOT NULL,
                    needs_verification INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES sources(source_id)
                );

                CREATE TABLE IF NOT EXISTS review_labels (
                    review_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    human_label TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES sources(source_id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                );
                """
            )

    def create_source(self, source_type, url=None, file_name=None):
        source_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sources (source_id, source_type, url, file_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, source_type, url, file_name, utc_now()),
            )
        return source_id

    def save_extracted_content(self, source_id, pieces):
        rows = []
        prepared = prepare_content_pieces(pieces)
        with self.connect() as conn:
            for piece in prepared:
                chunks = chunk_text(piece["text"])
                for index, chunk in enumerate(chunks):
                    content_id = str(uuid.uuid4())
                    row = {
                        "content_id": content_id,
                        "source_id": source_id,
                        "content_type": piece["content_type"],
                        "text": chunk,
                        "timestamp_sec": piece.get("timestamp_sec"),
                        "chunk_index": index,
                        "created_at": utc_now(),
                    }
                    conn.execute(
                        """
                        INSERT INTO extracted_content
                            (content_id, source_id, content_type, text, timestamp_sec, chunk_index, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["content_id"],
                            row["source_id"],
                            row["content_type"],
                            row["text"],
                            row["timestamp_sec"],
                            row["chunk_index"],
                            row["created_at"],
                        ),
                    )
                    rows.append(row)
        return rows

    def save_prediction(self, source_id, analysis, input_text):
        prediction_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (prediction_id, source_id, model_version, input_text, predicted_label,
                     confidence, final_status, needs_verification, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    source_id,
                    analysis["model_version"],
                    input_text,
                    analysis["label"],
                    float(analysis["confidence"]),
                    analysis["status"],
                    1 if analysis.get("needs_verification") else 0,
                    utc_now(),
                ),
            )
        return prediction_id

    def save_review_label(self, source_id, prediction_id, human_label, notes=None):
        human_label = (human_label or "").strip().lower()
        if human_label not in VALID_REVIEW_LABELS:
            raise ValueError(f"human_label must be one of: {', '.join(sorted(VALID_REVIEW_LABELS))}")

        review_id = str(uuid.uuid4())
        row = {
            "review_id": review_id,
            "source_id": source_id,
            "prediction_id": prediction_id,
            "human_label": human_label,
            "notes": notes,
            "created_at": utc_now(),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO review_labels
                    (review_id, source_id, prediction_id, human_label, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["review_id"],
                    row["source_id"],
                    row["prediction_id"],
                    row["human_label"],
                    row["notes"],
                    row["created_at"],
                ),
            )
        return row


class PineconeTextStore:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.namespace = os.getenv("PINECONE_NAMESPACE") or None
        self.embedding_provider = os.getenv("PINECONE_EMBEDDING_PROVIDER", "openai").strip().lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.pinecone_text_field = os.getenv("PINECONE_TEXT_FIELD", "text")
        self.batch_size = int(os.getenv("PINECONE_UPSERT_BATCH_SIZE", "32"))
        self.enabled = bool(self.api_key and self.index_name)
        if self.embedding_provider == "openai":
            self.enabled = self.enabled and bool(self.openai_api_key)
        self._openai_client = None
        self._pinecone_index = None
        missing = []
        if not self.api_key:
            missing.append("PINECONE_API_KEY")
        if not self.index_name:
            missing.append("PINECONE_INDEX_NAME")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if missing:
            logger.warning("Pinecone vector storage disabled; missing env values: %s", ", ".join(missing))
        else:
            logger.info(
                "Pinecone vector storage configured for index '%s' namespace '%s' provider '%s'",
                self.index_name,
                self.namespace or "__default__",
                self.embedding_provider,
            )

    def available(self):
        return self.enabled

    def status(self):
        status = {
            "enabled": self.enabled,
            "index": self.index_name,
            "namespace": self.namespace or "__default__",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "pinecone_text_field": self.pinecone_text_field,
            "batch_size": self.batch_size,
        }
        if not self._ensure_clients():
            status["connected"] = False
            return status
        try:
            stats = self._pinecone_index.describe_index_stats()
            status["connected"] = True
            status["stats"] = stats.to_dict() if hasattr(stats, "to_dict") else stats
            return status
        except Exception as exc:
            status["connected"] = False
            status["error"] = str(exc)
            return status

    def _ensure_clients(self):
        if not self.enabled:
            return False
        if self._openai_client and self._pinecone_index:
            return True

        try:
            from pinecone import Pinecone

            if self.embedding_provider == "openai":
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=self.openai_api_key)
            pinecone_client = Pinecone(api_key=self.api_key)
            self._pinecone_index = pinecone_client.Index(self.index_name)
            return True
        except Exception as exc:
            logger.warning("Pinecone/OpenAI vector storage disabled: %s", exc)
            self.enabled = False
            return False

    def upsert_content(self, source_id, source_type, pieces, prediction_id=None):
        if not self._ensure_clients():
            return {
                "enabled": False,
                "index": self.index_name,
                "namespace": self.namespace or "__default__",
                "upserted": 0,
            }

        records = []
        for piece in prepare_content_pieces(pieces):
            for chunk_index, chunk in enumerate(chunk_text(piece["text"])):
                records.append(
                    {
                        "id": f"{source_id}-{piece['content_type']}-{chunk_index}-{uuid.uuid4().hex[:8]}",
                        "text": chunk,
                        "metadata": {
                            "source_id": source_id,
                            "source_type": source_type,
                            "content_type": piece["content_type"],
                            "chunk_index": chunk_index,
                            "timestamp_sec": piece.get("timestamp_sec"),
                            "prediction_id": prediction_id,
                        },
                    }
                )

        if not records:
            return {
                "enabled": True,
                "index": self.index_name,
                "namespace": self.namespace or "__default__",
                "upserted": 0,
            }

        try:
            if self.embedding_provider == "pinecone":
                return self._upsert_with_pinecone_integrated_embedding(source_id, prediction_id, records)
            return self._upsert_with_openai_embeddings(source_id, prediction_id, records)
        except Exception as exc:
            logger.warning("Pinecone upsert failed; continuing without vector storage: %s", exc)
            return {
                "enabled": True,
                "index": self.index_name,
                "namespace": self.namespace or "__default__",
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "upserted": 0,
                "error": str(exc),
            }

    def _upsert_with_pinecone_integrated_embedding(self, source_id, prediction_id, records):
        namespace = self.namespace or "__default__"
        upserted = 0
        for start in range(0, len(records), min(self.batch_size, 96)):
            batch = records[start:start + min(self.batch_size, 96)]
            pinecone_records = []
            for record in batch:
                metadata = {key: value for key, value in record["metadata"].items() if value is not None}
                pinecone_records.append(
                    {
                        "_id": record["id"],
                        self.pinecone_text_field: record["text"],
                        **metadata,
                    }
                )
            self._pinecone_index.upsert_records(namespace=namespace, records=pinecone_records)
            upserted += len(pinecone_records)

        result = {
            "enabled": True,
            "index": self.index_name,
            "namespace": namespace,
            "embedding_provider": "pinecone",
            "pinecone_text_field": self.pinecone_text_field,
            "upserted": upserted,
        }
        logger.info(
            "Pinecone integrated upsert complete: source_id=%s prediction_id=%s index=%s namespace=%s records=%s",
            source_id,
            prediction_id,
            self.index_name,
            namespace,
            upserted,
        )
        return result

    def _upsert_with_openai_embeddings(self, source_id, prediction_id, records):
        try:
            upserted = 0
            for start in range(0, len(records), self.batch_size):
                batch = records[start:start + self.batch_size]
                embedding_response = self._openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=[record["text"] for record in batch],
                )
                vectors = []
                for record, item in zip(batch, embedding_response.data):
                    metadata = {key: value for key, value in record["metadata"].items() if value is not None}
                    metadata["text"] = record["text"]
                    vectors.append(
                        {
                            "id": record["id"],
                            "values": item.embedding,
                            "metadata": metadata,
                        }
                    )
                if self.namespace:
                    self._pinecone_index.upsert(vectors=vectors, namespace=self.namespace)
                else:
                    self._pinecone_index.upsert(vectors=vectors)
                upserted += len(vectors)

            result = {
                "enabled": True,
                "index": self.index_name,
                "namespace": self.namespace or "__default__",
                "embedding_provider": "openai",
                "embedding_model": self.embedding_model,
                "upserted": upserted,
            }
            logger.info(
                "Pinecone upsert complete: source_id=%s prediction_id=%s index=%s namespace=%s vectors=%s",
                source_id,
                prediction_id,
                self.index_name,
                self.namespace or "__default__",
                upserted,
            )
            return result
        except Exception as exc:
            raise exc
