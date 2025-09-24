document.addEventListener('DOMContentLoaded', () => {
    // --- State Management ---
    const state = {
        coins: [],
        watchlist: [],
        currentView: 'all', // 'all' or 'watchlist'
        selectedCoinId: null,
        chartInstance: null,
    };

    // --- DOM Elements ---
    const elements = {
        container: document.querySelector('.market-grid'),
        coinList: document.getElementById('coin-list'),
        allCoinsTab: document.getElementById('all-coins-tab'),
        watchlistTab: document.getElementById('watchlist-tab'),
        searchInput: document.getElementById('search-input'),
        selectedCoin: {
            icon: document.getElementById('selected-coin-icon'),
            name: document.getElementById('selected-coin-name'),
            symbol: document.getElementById('selected-coin-symbol'),
            price: document.getElementById('selected-coin-price'),
            change: document.getElementById('selected-coin-change'),
            toggleWatchlistBtn: document.getElementById('toggle-watchlist-btn'),
        },
        chartCanvas: document.getElementById('priceChart'),
        orderBookContent: document.getElementById('orders-content'),
        infoContent: document.getElementById('news-content'),
    };

    // --- API URLs from data attributes ---
    const API = {
        coins: elements.container.dataset.apiCoins,
        watchlist: elements.container.dataset.apiWatchlist,
        toggleWatchlist: elements.container.dataset.apiToggleWatchlist,
        coinDetails: elements.container.dataset.apiCoinDetails,
    };

    // --- Helper Functions ---
    const formatPrice = (price) => {
        if (price === null || price === undefined) return 'N/A';
        return price.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 6 });
    };

    const formatPercentage = (change) => {
        if (change === null || change === undefined) return '';
        const formatted = change.toFixed(2) + '%';
        const colorClass = change >= 0 ? 'text-success' : 'text-danger';
        return `<span class="${colorClass}">${formatted}</span>`;
    };

    // --- API Fetcher ---
    const fetchData = async (url, options = {}) => {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error(`Fetch error for ${url}:`, error);
            // Here you could show a user-facing error message
            return { success: false, error: error.message };
        }
    };

    // --- Rendering ---
    const renderCoinList = () => {
        const listToRender = state.currentView === 'all' ? state.coins : state.watchlist;
        const query = elements.searchInput.value.toLowerCase();
        const filteredList = listToRender.filter(c => c.name.toLowerCase().includes(query) || c.symbol.toLowerCase().includes(query));

        elements.coinList.innerHTML = ''; // Clear list
        if (filteredList.length === 0) {
            elements.coinList.innerHTML = '<li class="list-group-item bg-dark text-muted text-center">Ничего не найдено</li>';
            return;
        }

        filteredList.forEach(coin => {
            const li = document.createElement('li');
            li.className = `list-group-item list-group-item-action bg-dark text-light border-secondary cursor-pointer ${state.selectedCoinId === coin.id ? 'active' : ''}`;
            li.dataset.coinId = coin.id;
            li.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div class="d-flex align-items-center">
                        <img src="${coin.image}" width="24" height="24" class="me-2">
                        <strong>${coin.symbol.toUpperCase()}</strong>
                    </div>
                    <small>${formatPrice(coin.current_price)}</small>
                    <small>${formatPercentage(coin.price_change_percentage_24h)}</small>
                </div>
            `;
            elements.coinList.appendChild(li);
        });
    };

    const updateSelectedCoinView = (coinData) => {
        const { info, history, tickers } = coinData;
        elements.selectedCoin.icon.src = info.image.large;
        elements.selectedCoin.name.textContent = info.name;
        elements.selectedCoin.symbol.textContent = info.symbol.toUpperCase();
        elements.selectedCoin.price.textContent = formatPrice(info.market_data.current_price.usd);
        elements.selectedCoin.change.innerHTML = formatPercentage(info.market_data.price_change_percentage_24h);
        updateWatchlistButton();
        renderPriceChart(history);
        renderOrderBook(tickers);
        renderInfoTab(info);
    };

    const updateWatchlistButton = () => {
        const isWatchlisted = state.watchlist.some(c => c.id === state.selectedCoinId);
        elements.selectedCoin.toggleWatchlistBtn.classList.toggle('active', isWatchlisted);
    };

    const renderPriceChart = (history) => {
        const ctx = elements.chartCanvas.getContext('2d');
        const labels = history.prices.map(p => new Date(p[0]).toLocaleDateString());
        const data = history.prices.map(p => p[1]);

        if (state.chartInstance) state.chartInstance.destroy();

        state.chartInstance = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets: [{ label: 'Цена (USD)', data, borderColor: '#17a2b8', backgroundColor: 'rgba(23, 162, 184, 0.1)', fill: true, tension: 0.1, pointRadius: 0 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { beginAtZero: false } } }
        });
    };

    const renderOrderBook = (tickers) => {
        let html = '<ul class="list-group list-group-flush">';
        tickers.tickers.slice(0, 20).forEach(t => {
            html += `<li class="list-group-item bg-dark text-light border-secondary d-flex justify-content-between"><span>${t.market.name}</span> <span>${formatPrice(t.last)}</span></li>`;
        });
        elements.orderBookContent.innerHTML = html + '</ul>';
    };

    const renderInfoTab = (info) => {
        elements.infoContent.innerHTML = `<h5>О ${info.name}</h5><p>${info.description.en || 'Описание недоступно.'}</p>`;
    };

    // --- Logic ---
    const loadInitialData = async () => {
        const [coinsResult, watchlistResult] = await Promise.all([fetchData(API.coins), fetchData(API.watchlist)]);
        if (coinsResult.success) state.coins = coinsResult.data;
        if (watchlistResult.success) state.watchlist = watchlistResult.data;
        
        renderCoinList();
        if (state.coins.length > 0) {
            selectCoin(state.coins[0].id);
        }
    };

    const selectCoin = async (coinId) => {
        if (!coinId) return;
        state.selectedCoinId = coinId;
        const url = API.coinDetails.replace('__COIN_ID__', coinId);
        const result = await fetchData(url);
        if (result.success) {
            updateSelectedCoinView(result.data);
        }
        // Highlight in list
        document.querySelectorAll('#coin-list li').forEach(li => li.classList.remove('active'));
        document.querySelector(`#coin-list li[data-coin-id="${coinId}"]`)?.classList.add('active');
    };

    const toggleWatchlist = async () => {
        if (!state.selectedCoinId) return;
        const result = await fetchData(API.toggleWatchlist, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').getAttribute('content') },
            body: JSON.stringify({ coin_id: state.selectedCoinId })
        });

        if (result.success) {
            // Refetch watchlist to get the latest state
            const watchlistResult = await fetchData(API.watchlist);
            if (watchlistResult.success) state.watchlist = watchlistResult.data;
            updateWatchlistButton();
            if (state.currentView === 'watchlist') renderCoinList();
        }
    };

    // --- Event Listeners ---
    elements.allCoinsTab.addEventListener('click', () => {
        state.currentView = 'all';
        elements.allCoinsTab.classList.add('active');
        elements.watchlistTab.classList.remove('active');
        renderCoinList();
    });

    elements.watchlistTab.addEventListener('click', () => {
        state.currentView = 'watchlist';
        elements.watchlistTab.classList.add('active');
        elements.allCoinsTab.classList.remove('active');
        renderCoinList();
    });

    elements.searchInput.addEventListener('input', renderCoinList);

    elements.coinList.addEventListener('click', (e) => {
        const coinItem = e.target.closest('[data-coin-id]');
        if (coinItem) selectCoin(coinItem.dataset.coinId);
    });

    elements.selectedCoin.toggleWatchlistBtn.addEventListener('click', toggleWatchlist);

    // --- Init ---
    loadInitialData();
});
