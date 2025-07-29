/**
 * FAQ Suggestions Component
 * Displays clickable suggested questions for the chat interface
 */

class SuggestedQuestions {
    constructor(containerId, onQuestionSelect) {
        this.containerId = containerId;
        this.onQuestionSelect = onQuestionSelect;
        this.faqData = [];
        this.isVisible = false;
        this.init();
        this.setupScrollObserver();
    }

    async init() {
        try {
            await this.loadFAQData();
            this.render();
        } catch (error) {
            console.error('Error initializing SuggestedQuestions:', error);
        }
    }

    async loadFAQData() {
        try {
            const response = await fetch('/static/faq_data.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.faqData = await response.json();
        } catch (error) {
            console.error('Error loading FAQ data:', error);
            this.faqData = [];
        }
    }

    render() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container with id '${this.containerId}' not found`);
            return;
        }

        if (!this.isVisible) {
            container.innerHTML = '';
            return;
        }

        if (this.faqData.length === 0) {
            container.innerHTML = `
                <div class="suggested-questions-loading">
                    Memuat pertanyaan...
                </div>
            `;
            return;
        }

        const html = `
            <div class="suggested-questions-container">
                <div class="suggested-questions-header">
                    <h3 class="text-lg font-semibold text-gray-700 mb-3">
                        💡 Pertanyaan yang Sering Diajukan
                    </h3>
                    <p class="text-sm text-gray-500 mb-4">
                        Klik pertanyaan di bawah untuk mendapatkan jawaban instan
                    </p>
                </div>
                <div class="suggested-questions-grid">
                    ${this.faqData.map((faq, index) => `
                        <button 
                            class="suggested-question-btn"
                            data-index="${index}"
                            onclick="suggestedQuestions.handleQuestionClick(${index})"
                        >
                            <div class="question-text">${index + 1}. ${faq.question}</div>
                            <div class="question-icon">💬</div>
                        </button>
                    `).join('')}
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    handleQuestionClick(index) {
        const faqItem = this.faqData[index];
        if (faqItem && this.onQuestionSelect) {
            this.onQuestionSelect(faqItem);
        }
    }

    show() {
        this.isVisible = true;
        this.render();
        
        // Trigger auto-scroll after rendering
        setTimeout(() => {
            this.scrollToSuggestions();
        }, 100);
    }

    hide() {
        this.isVisible = false;
        this.render();
    }

    toggle() {
        this.isVisible = !this.isVisible;
        this.render();
    }

    isShown() {
        return this.isVisible;
    }
    
    scrollToSuggestions() {
        const container = document.getElementById(this.containerId);
        if (container && this.isVisible) {
            // Scroll to the suggestions container
            container.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
            
            // Also ensure the main chat container scrolls to show the suggestions
            const mainContainer = document.querySelector('.max-w-xl.mx-auto.flex.flex-col');
            if (mainContainer) {
                setTimeout(() => {
                    mainContainer.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'end' 
                    });
                }, 100);
            }
        }
    }
    
    setupScrollObserver() {
        // Watch for changes in the container to trigger scroll
        const container = document.getElementById(this.containerId);
        if (container && window.MutationObserver) {
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'childList' && this.isVisible) {
                        // Content changed and suggestions are visible, trigger scroll
                        setTimeout(() => {
                            this.scrollToSuggestions();
                        }, 50);
                    }
                });
            });
            
            observer.observe(container, {
                childList: true,
                subtree: true
            });
        }
    }
}

// Global instance for easy access
let suggestedQuestions = null; 