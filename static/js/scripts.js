document.addEventListener('DOMContentLoaded', () => {
    initializeTooltips();
    setupRealTimePriceUpdates();
    enableFormValidation();
});

/**
 * Инициализация Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltips].forEach(el => new bootstrap.Tooltip(el));
}

/**
 * Обновление цены BTC в режиме реального времени на трейдер-дэшборде
 */
function setupRealTimePriceUpdates() {
    const dashboard = document.getElementById('traderDashboard');
    const priceEl = document.getElementById('btcPrice');
    const changeEl = document.getElementById('btcChange');

    if (!dashboard || !priceEl || !changeEl) return;

    const updatePrice = async () => {
        try {
            const response = await fetch('/api/crypto/BTC-USD');
            const data = await response.json();

            if (data && !data.error) {
                priceEl.textContent = `$${data.current_price.toFixed(2)}`;
                changeEl.textContent = `${data.change.toFixed(2)}%`;
                changeEl.className = data.change >= 0 ? 'text-success' : 'text-danger';
            }
        } catch (err) {
            console.warn('Ошибка получения цены BTC:', err);
        }
    };

    updatePrice();
    setInterval(updatePrice, 5000);
}

/**
 * Включение проверки формы с Bootstrap валидацией
 */
function enableFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}
