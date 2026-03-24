class AudioProcessor {
    constructor(socket) {
        this.socket = socket;
        this.audioContext = null;
        this.mediaStream = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.analyzerNode = null;
        this.dataArray = null;
        this.bufferLength = 0;
    }

    async setupAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyzerNode = this.audioContext.createAnalyser();
            this.analyzerNode.fftSize = 2048;
            this.bufferLength = this.analyzerNode.frequencyBinCount;
            this.dataArray = new Uint8Array(this.bufferLength);
            return true;
        } catch (error) {
            console.error('Failed to setup audio context:', error);
            return false;
        }
    }

    async startRecording() {
        if (!this.audioContext) {
            await this.setupAudioContext();
        }

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.analyzerNode);

            this.mediaRecorder = new MediaRecorder(this.mediaStream);
            this.audioChunks = [];
            this.isRecording = true;

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    this.processAudioChunk(event.data);
                }
            };

            this.mediaRecorder.start(1000); // Collect data every second
            this.startVisualization();
            return true;
        } catch (error) {
            console.error('Failed to start recording:', error);
            return false;
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            return new Promise((resolve) => {
                this.mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                    this.isRecording = false;
                    this.mediaStream.getTracks().forEach(track => track.stop());
                    this.stopVisualization();
                    resolve(audioBlob);
                };
                this.mediaRecorder.stop();
            });
        }
        return Promise.resolve(null);
    }

    async processAudioChunk(chunk) {
        try {
            const arrayBuffer = await chunk.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            const analysis = this.analyzeAudioBuffer(audioBuffer);

            if (this.socket) {
                this.socket.emit('audio_analysis', analysis);
            }

            return analysis;
        } catch (error) {
            console.error('Error processing audio chunk:', error);
            return null;
        }
    }

    analyzeAudioBuffer(audioBuffer) {
        const channelData = audioBuffer.getChannelData(0);
        const samples = channelData.length;

        let sum = 0;
        let maxAmplitude = 0;
        let minAmplitude = 0;

        for (let i = 0; i < samples; i++) {
            const amplitude = channelData[i];
            sum += Math.abs(amplitude);
            maxAmplitude = Math.max(maxAmplitude, amplitude);
            minAmplitude = Math.min(minAmplitude, amplitude);
        }

        return {
            averageAmplitude: sum / samples,
            maxAmplitude: maxAmplitude,
            minAmplitude: minAmplitude,
            duration: audioBuffer.duration,
            sampleRate: audioBuffer.sampleRate,
            timestamp: Date.now()
        };
    }

    startVisualization() {
        const canvas = document.getElementById('audioVisualizer');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const WIDTH = canvas.width;
        const HEIGHT = canvas.height;

        const drawVisualizer = () => {
            if (!this.isRecording) return;

            requestAnimationFrame(drawVisualizer);
            this.analyzerNode.getByteTimeDomainData(this.dataArray);

            ctx.fillStyle = 'rgb(200, 200, 200)';
            ctx.fillRect(0, 0, WIDTH, HEIGHT);
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'rgb(0, 0, 0)';
            ctx.beginPath();

            const sliceWidth = WIDTH * 1.0 / this.bufferLength;
            let x = 0;

            for (let i = 0; i < this.bufferLength; i++) {
                const v = this.dataArray[i] / 128.0;
                const y = v * HEIGHT / 2;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }

                x += sliceWidth;
            }

            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        };

        drawVisualizer();
    }

    stopVisualization() {
        // Clear visualization if needed
        const canvas = document.getElementById('audioVisualizer');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
}
