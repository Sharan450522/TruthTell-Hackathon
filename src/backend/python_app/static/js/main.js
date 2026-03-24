document.addEventListener('DOMContentLoaded', function() {
    // Initialize Socket.IO
    const socket = io();

    // Initialize AudioProcessor
    const audioProcessor = new AudioProcessor();

    // Form elements
    const form = document.getElementById('transcriptionForm');
    const videoTypeSelect = document.getElementById('videoType');
    const urlContainer = document.getElementById('urlContainer');
    const textContainer = document.getElementById('textContainer');
    const startButton = document.getElementById('startTranscription');
    const stopButton = document.getElementById('stopTranscription');
    const loadingMessage = document.getElementById('loadingMessage');

    // Results containers
    const transcriptionResult = document.getElementById('transcriptionResult');
    const factCheckResult = document.getElementById('factCheckResult');
    const confidenceChart = document.getElementById('confidenceChart');

    // Initialize audio processor
    audioProcessor.initialize().catch(error => {
        console.error('Audio processor initialization failed:', error);
    });

    // Toggle input type based on content type selection
    videoTypeSelect.addEventListener('change', function() {
        const selectedType = this.value;
        urlContainer.classList.toggle('hidden', selectedType === 'text');
        textContainer.classList.toggle('hidden', selectedType !== 'text');
        startButton.querySelector('#startButtonText').textContent = selectedType === 'live' ? 'Start Stream' : 'Start Analysis';
        stopButton.classList.toggle('hidden', selectedType !== 'live');
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const contentType = videoTypeSelect.value;
        const language = document.getElementById('language').value;

        if (!validateInput(contentType)) return;

        startButton.disabled = true;
        loadingMessage.classList.remove('hidden');

        try {
            if (contentType === 'text') {
                const text = document.getElementById('textInput').value;
                await handleTextAnalysis(text, language);
            } else {
                const url = document.getElementById('videoUrl').value;
                if (contentType === 'live') {
                    handleLiveStream(url, language);
                } else {
                    await handleRecordedVideo(url, language);
                }
            }
        } catch (error) {
            showError('An error occurred during analysis: ' + error.message);
        }
    });

    // Handle live stream stop
    stopButton.addEventListener('click', async function() {
        try {
            const audioBlob = await audioProcessor.stopRecording();
            if (audioBlob) {
                socket.emit('audio_data', audioBlob);
            }
            socket.emit('stop_stream');
            stopButton.classList.add('hidden');
            startButton.disabled = false;
            loadingMessage.classList.add('hidden');
        } catch (error) {
            showError('Error stopping stream: ' + error.message);
        }
    });

    // Socket.IO event handlers
    socket.on('transcription_update', function(data) {
        updateTranscriptionResult(data.text);
    });

    socket.on('fact_check_update', function(data) {
        updateFactCheckResult(data.facts);
    });

    socket.on('confidence_update', function(data) {
        updateConfidenceChart(data.confidence);
    });

    socket.on('analysis_complete', function() {
        startButton.disabled = false;
        loadingMessage.classList.add('hidden');
        if (videoTypeSelect.value === 'live') {
            stopButton.classList.remove('hidden');
        }
    });

    socket.on('error', function(data) {
        showError(data.message);
    });

    // Helper functions
    function validateInput(contentType) {
        if (contentType === 'text') {
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                showError('Please enter some text to analyze');
                return false;
            }
        } else {
            const url = document.getElementById('videoUrl').value.trim();
            if (!utils.validateUrl(url)) {
                showError('Please enter a valid URL');
                return false;
            }
        }
        return true;
    }

    async function handleTextAnalysis(text, language) {
        try {
            const response = await fetch('/analyze/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text, language })
            });

            if (!response.ok) {
                throw new Error('Text analysis failed');
            }

            const result = await response.json();
            updateResults(result);
        } catch (error) {
            throw new Error('Text analysis failed: ' + error.message);
        } finally {
            startButton.disabled = false;
            loadingMessage.classList.add('hidden');
        }
    }

    async function handleRecordedVideo(url, language) {
        try {
            const response = await fetch('/analyze/video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url, language })
            });

            if (!response.ok) {
                throw new Error('Video analysis failed');
            }

            const result = await response.json();
            updateResults(result);
        } catch (error) {
            throw new Error('Video analysis failed: ' + error.message);
        } finally {
            startButton.disabled = false;
            loadingMessage.classList.add('hidden');
        }
    }

    async function handleLiveStream(url, language) {
        try {
            await audioProcessor.startRecording();
            socket.emit('start_stream', { url, language });
        } catch (error) {
            showError('Failed to start live stream: ' + error.message);
            startButton.disabled = false;
            loadingMessage.classList.add('hidden');
        }
    }

    function updateResults(result) {
        updateTranscriptionResult(result.transcription);
        updateFactCheckResult(result.facts);
        updateConfidenceChart(result.confidence);
    }

    function updateTranscriptionResult(text) {
        transcriptionResult.innerHTML = `<p>${text}</p>`;
    }

    function updateFactCheckResult(facts) {
        const factsHtml = facts.map(fact => `
            <div class="mb-2">
                <span class="${fact.verified ? 'text-green-600' : 'text-red-600'}">
                    ${fact.verified ? '✓' : '✗'}
                </span>
                ${fact.statement}
            </div>
        `).join('');
        factCheckResult.innerHTML = factsHtml;
    }

    function updateConfidenceChart(confidence) {
        confidenceChart.innerHTML = `
            <div class="flex items-center">
                <div class="w-full bg-gray-200 rounded-full h-2.5">
                    <div class="bg-blue-600 h-2.5 rounded-full" style="width: ${confidence}%"></div>
                </div>
                <span class="ml-2">${confidence}%</span>
            </div>
        `;
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);

        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    // Initialize news handler
    const newsHandler = new NewsHandler();
});
