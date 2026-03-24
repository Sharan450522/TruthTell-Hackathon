class NewsHandler {
    constructor() {
        this.newsContainer = document.getElementById('trendingNews');
        this.refreshButton = document.getElementById('refreshNews');
        this.lastUpdate = null;
        this.isLoading = false;

        this.initialize();
    }

    initialize() {
        this.refreshButton.addEventListener('click', () => this.refreshNews());
        this.loadInitialNews();

        // Auto-refresh every 5 minutes
        setInterval(() => this.loadInitialNews(), 300000);
    }

    async loadInitialNews() {
        if (this.isLoading) return;

        try {
            this.isLoading = true;
            this.showLoadingIndicator();

            const response = await fetch('/api/news/initial');
            if (!response.ok) throw new Error('Failed to fetch initial news');

            const news = await response.json();
            this.updateNewsDisplay(news, true);
            this.lastUpdate = new Date();
        } catch (error) {
            this.showError('Failed to load initial news');
            console.error('Error loading initial news:', error);
        } finally {
            this.hideLoadingIndicator();
            this.isLoading = false;
        }
    }

    async loadMoreNews() {
        if (this.isLoading) return;

        try {
            this.isLoading = true;
            const lastNewsId = this.getLastNewsId();

            const response = await fetch(`/api/news/more?lastId=${lastNewsId}`);
            if (!response.ok) throw new Error('Failed to fetch more news');

            const news = await response.json();
            if (news.length > 0) {
                this.appendNewsItems(news);
            }
        } catch (error) {
            console.error('Error loading more news:', error);
        } finally {
            this.isLoading = false;
        }
    }

    async refreshNews(silent = false) {
        if (this.isLoading) return;

        try {
            this.isLoading = true;
            if (!silent) {
                this.refreshButton.disabled = true;
                this.showLoadingIndicator();
            }

            const response = await fetch('/api/news/latest');
            if (!response.ok) throw new Error('Failed to fetch latest news');

            const news = await response.json();
            if (news.length > 0) {
                this.updateNewsDisplay(news, false);
                this.lastUpdate = new Date();
            }
        } catch (error) {
            if (!silent) {
                this.showError('Failed to refresh news');
            }
            console.error('Error refreshing news:', error);
        } finally {
            if (!silent) {
                this.hideLoadingIndicator();
                this.refreshButton.disabled = false;
            }
            this.isLoading = false;
        }
    }

    updateNewsDisplay(news, reset = false) {
        if (reset) {
            this.newsContainer.innerHTML = '';
            this.newsCache.clear();
        }

        this.appendNewsItems(news);
    }

    appendNewsItems(news) {
        const fragment = document.createDocumentFragment();

        news.forEach(item => {
            if (!this.newsCache.has(item.id)) {
                const newsElement = this.createNewsElement(item);
                fragment.appendChild(newsElement);
                this.newsCache.set(item.id, item);
            }
        });

        this.newsContainer.appendChild(fragment);
        this.trimOldNews();
    }

    createNewsElement(item) {
        const div = document.createElement('div');
        div.className = 'news-item border-b border-gray-200 pb-3';
        div.dataset.newsId = item.id;

        div.innerHTML = `
            <h3 class="font-medium text-gray-900 mb-1">${item.title}</h3>
            <p class="text-gray-600 text-sm mb-2">${this.truncateText(item.description, 150)}</p>
            <div class="flex justify-between items-center text-xs text-gray-500">
                <span>${this.formatTimestamp(item.timestamp)}</span>
                <div class="flex items-center gap-2">
                    <button class="share-btn text-blue-600 hover:text-blue-800">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684 3 3 0 00-5.367-2.684 3 3 0 005.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z">
                            </path>
                        </svg>
                    </button>
                    <a href="${item.url}" target="_blank" rel="noopener noreferrer"
                       class="text-blue-600 hover:text-blue-800">Read more</a>
                </div>
            </div>
        `;

        // Add share button functionality
        div.querySelector('.share-btn').addEventListener('click', () => {
            this.shareNews(item);
        });

        return div;
    }

    trimOldNews() {
        const newsItems = this.newsContainer.children;
        if (newsItems.length > this.maxNewsItems) {
            for (let i = this.maxNewsItems; i < newsItems.length; i++) {
                const item = newsItems[i];
                this.newsCache.delete(item.dataset.newsId);
                item.remove();
            }
        }
    }

    async shareNews(newsItem) {
        if (navigator.share) {
            try {
                await navigator.share({
                    title: newsItem.title,
                    text: newsItem.description,
                    url: newsItem.url
                });
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Error sharing news:', error);
                }
            }
        } else {
            // Fallback: Copy link to clipboard
            navigator.clipboard.writeText(newsItem.url)
                .then(() => this.showToast('Link copied to clipboard'))
                .catch(error => console.error('Failed to copy link:', error));
        }
    }

    getLastNewsId() {
        const lastItem = this.newsContainer.lastElementChild;
        return lastItem ? lastItem.dataset.newsId : null;
    }

    showLoadingIndicator() {
        const loader = document.createElement('div');
        loader.id = 'newsLoadingIndicator';
        loader.className = 'flex justify-center items-center py-4';
        loader.innerHTML = `
            <svg class="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
        `;
        this.newsContainer.appendChild(loader);
    }

    hideLoadingIndicator() {
        const loader = document.getElementById('newsLoadingIndicator');
        if (loader) {
            loader.remove();
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        this.newsContainer.prepend(errorDiv);

        setTimeout(() => errorDiv.remove(), 5000);
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateY(0)';
        });

        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.transform = 'translateY(100%)';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 3000);
    }

    truncateText(text, maxLength = 100) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMinutes = Math.floor((now - date) / (1000 * 60));

        if (diffMinutes < 1) return 'Just now';
        if (diffMinutes < 60) return `${diffMinutes}m ago`;

        const diffHours = Math.floor(diffMinutes / 60);
        if (diffHours < 24) return `${diffHours}h ago`;

        const diffDays = Math.floor(diffHours / 24);
        if (diffDays < 7) return `${diffDays}d ago`;

        return date.toLocaleDateString();
    }
}
