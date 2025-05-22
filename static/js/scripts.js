document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Real-time price updates for trader dashboard
    if (document.getElementById('traderDashboard')) {
        setInterval(function() {
            fetch('/api/crypto/BTC-USD')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        document.getElementById('btcPrice').innerText = '$' + data.current_price.toFixed(2);
                        document.getElementById('btcChange').innerText = data.change.toFixed(2) + '%';
                        document.getElementById('btcChange').className = data.change >= 0 ? 'text-success' : 'text-danger';
                    }
                });
        }, 5000);
    }

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
});