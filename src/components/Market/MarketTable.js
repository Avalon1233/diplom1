import React, { useState, useMemo } from 'react';

const MarketTable = ({ data, onAddToWatchlist, onSetAlert, onExport, loading }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [filter, setFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  const getCryptoIcon = (symbol) => {
    if (symbol.includes('BTC')) return 'bi-currency-bitcoin text-warning';
    if (symbol.includes('ETH')) return 'bi-ethereum text-info';
    if (symbol.includes('BNB')) return 'bi-coin text-warning';
    if (symbol.includes('ADA')) return 'bi-coin text-primary';
    if (symbol.includes('SOL')) return 'bi-coin text-purple';
    return 'bi-coin text-secondary';
  };

  const getCryptoName = (symbol) => {
    if (symbol.includes('BTC')) return 'Bitcoin';
    if (symbol.includes('ETH')) return 'Ethereum';
    if (symbol.includes('BNB')) return 'Binance Coin';
    if (symbol.includes('ADA')) return 'Cardano';
    if (symbol.includes('SOL')) return 'Solana';
    return symbol.split('-')[0];
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: price < 1 ? 4 : 2
    }).format(price);
  };

  const formatVolume = (volume) => {
    if (volume >= 1e9) {
      return `${(volume / 1e9).toFixed(1)}B`;
    } else if (volume >= 1e6) {
      return `${(volume / 1e6).toFixed(1)}M`;
    } else if (volume >= 1e3) {
      return `${(volume / 1e3).toFixed(1)}K`;
    }
    return new Intl.NumberFormat('en-US').format(Math.round(volume));
  };

  const formatChange = (change) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}%`;
  };

  // Filter and search data
  const filteredData = useMemo(() => {
    let filtered = data;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(item =>
        item.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        getCryptoName(item.symbol).toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply change filter
    if (filter === 'up') {
      filtered = filtered.filter(item => item.change >= 0);
    } else if (filter === 'down') {
      filtered = filtered.filter(item => item.change < 0);
    } else if (filter === 'high_volume') {
      const avgVolume = data.reduce((sum, item) => sum + item.volume, 0) / data.length;
      filtered = filtered.filter(item => item.volume > avgVolume);
    }

    return filtered;
  }, [data, searchTerm, filter]);

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortConfig.key) return filteredData;

    return [...filteredData].sort((a, b) => {
      let aValue = a[sortConfig.key];
      let bValue = b[sortConfig.key];

      // Handle string sorting for symbol
      if (sortConfig.key === 'symbol') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [filteredData, sortConfig]);

  // Paginate data
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return sortedData.slice(startIndex, startIndex + itemsPerPage);
  }, [sortedData, currentPage]);

  const totalPages = Math.ceil(sortedData.length / itemsPerPage);

  const handleSort = (key) => {
    setSortConfig(prevConfig => ({
      key,
      direction: prevConfig.key === key && prevConfig.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const getSortIcon = (key) => {
    if (sortConfig.key !== key) {
      return 'bi-arrow-up-down';
    }
    return sortConfig.direction === 'asc' ? 'bi-arrow-up' : 'bi-arrow-down';
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
  };

  const renderPagination = () => {
    const pages = [];
    const maxVisiblePages = 5;
    
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <li key={i} className={`page-item ${i === currentPage ? 'active' : ''}`}>
          <button className="page-link" onClick={() => handlePageChange(i)}>
            {i}
          </button>
        </li>
      );
    }

    return (
      <nav>
        <ul className="pagination pagination-sm mb-0">
          <li className={`page-item ${currentPage === 1 ? 'disabled' : ''}`}>
            <button 
              className="page-link" 
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              Предыдущая
            </button>
          </li>
          {pages}
          <li className={`page-item ${currentPage === totalPages ? 'disabled' : ''}`}>
            <button 
              className="page-link" 
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              Следующая
            </button>
          </li>
        </ul>
      </nav>
    );
  };

  return (
    <div className="card shadow-sm border-0 market-table-container">
      <div className="card-header bg-white border-0 py-3">
        <div className="row align-items-center">
          <div className="col-md-6">
            <h5 className="mb-0">
              <i className="bi bi-table text-primary me-2"></i>
              Рыночные данные
            </h5>
          </div>
          <div className="col-md-6">
            <div className="d-flex align-items-center justify-content-md-end">
              {/* Search */}
              <div className="input-group input-group-sm me-3" style={{ width: '250px' }}>
                <span className="input-group-text bg-light border-end-0">
                  <i className="bi bi-search"></i>
                </span>
                <input
                  type="text"
                  className="form-control border-start-0"
                  placeholder="Поиск криптовалюты..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>

              {/* Filters */}
              <div className="dropdown me-2">
                <button
                  className="btn btn-sm btn-outline-secondary dropdown-toggle"
                  type="button"
                  data-bs-toggle="dropdown"
                >
                  <i className="bi bi-funnel"></i> Фильтр
                </button>
                <ul className="dropdown-menu">
                  <li>
                    <button
                      className="dropdown-item"
                      onClick={() => setFilter('all')}
                    >
                      <i className="bi bi-list me-2"></i>Все
                    </button>
                  </li>
                  <li>
                    <button
                      className="dropdown-item"
                      onClick={() => setFilter('up')}
                    >
                      <i className="bi bi-arrow-up text-success me-2"></i>Растущие
                    </button>
                  </li>
                  <li>
                    <button
                      className="dropdown-item"
                      onClick={() => setFilter('down')}
                    >
                      <i className="bi bi-arrow-down text-danger me-2"></i>Падающие
                    </button>
                  </li>
                  <li>
                    <button
                      className="dropdown-item"
                      onClick={() => setFilter('high_volume')}
                    >
                      <i className="bi bi-bar-chart text-info me-2"></i>Высокий объём
                    </button>
                  </li>
                </ul>
              </div>

              {/* Export */}
              <button 
                className="btn btn-sm btn-outline-secondary"
                onClick={onExport}
              >
                <i className="bi bi-download"></i>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="card-body p-0">
        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
          </div>
        )}

        <div className="table-responsive">
          <table className="table table-hover mb-0 market-table">
            <thead className="bg-light">
              <tr>
                <th className="border-0 px-4 py-3">
                  <div
                    className="sortable-header"
                    onClick={() => handleSort('symbol')}
                  >
                    Криптовалюта
                    <i className={`bi ${getSortIcon('symbol')} sort-icon`}></i>
                  </div>
                </th>
                <th className="border-0 px-4 py-3 text-end">
                  <div
                    className="sortable-header justify-content-end"
                    onClick={() => handleSort('price')}
                  >
                    Цена
                    <i className={`bi ${getSortIcon('price')} sort-icon`}></i>
                  </div>
                </th>
                <th className="border-0 px-4 py-3 text-end">
                  <div
                    className="sortable-header justify-content-end"
                    onClick={() => handleSort('change')}
                  >
                    Изменение 24ч
                    <i className={`bi ${getSortIcon('change')} sort-icon`}></i>
                  </div>
                </th>
                <th className="border-0 px-4 py-3 text-end">Максимум 24ч</th>
                <th className="border-0 px-4 py-3 text-end">Минимум 24ч</th>
                <th className="border-0 px-4 py-3 text-end">
                  <div
                    className="sortable-header justify-content-end"
                    onClick={() => handleSort('volume')}
                  >
                    Объём
                    <i className={`bi ${getSortIcon('volume')} sort-icon`}></i>
                  </div>
                </th>
                <th className="border-0 px-4 py-3 text-center">Действия</th>
              </tr>
            </thead>
            <tbody>
              {paginatedData.map((crypto) => {
                const isPositive = crypto.change >= 0;
                return (
                  <tr key={crypto.symbol} className="market-row">
                    <td className="px-4 py-3">
                      <div className="d-flex align-items-center">
                        <div className="crypto-icon me-3">
                          <i className={`bi ${getCryptoIcon(crypto.symbol)} fs-4`}></i>
                        </div>
                        <div>
                          <div className="fw-bold">
                            {crypto.symbol.replace('-', ' / ')}
                          </div>
                          <small className="text-muted">
                            {getCryptoName(crypto.symbol)}
                          </small>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-end">
                      <span className="fw-bold">{formatPrice(crypto.price)}</span>
                    </td>
                    <td className="px-4 py-3 text-end">
                      <div className="d-flex align-items-center justify-content-end">
                        <span
                          className={`badge ${
                            isPositive
                              ? 'bg-success bg-opacity-10 text-success'
                              : 'bg-danger bg-opacity-10 text-danger'
                          } px-2 py-1 me-2`}
                        >
                          <i
                            className={`bi bi-triangle-fill me-1`}
                            style={{
                              transform: isPositive ? 'rotate(0deg)' : 'rotate(180deg)',
                              fontSize: '0.6rem'
                            }}
                          ></i>
                          {formatChange(crypto.change)}
                        </span>
                        <div className="progress" style={{ width: '40px', height: '4px' }}>
                          <div
                            className={`progress-bar ${
                              isPositive ? 'bg-success' : 'bg-danger'
                            }`}
                            style={{
                              width: `${Math.min(Math.abs(crypto.change) * 5, 100)}%`
                            }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-end text-muted">
                      {formatPrice(crypto.high)}
                    </td>
                    <td className="px-4 py-3 text-end text-muted">
                      {formatPrice(crypto.low)}
                    </td>
                    <td className="px-4 py-3 text-end">
                      <span className="text-muted">{formatVolume(crypto.volume)}</span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <div className="btn-group btn-group-sm" role="group">
                        <a
                          href={`/trader/chart/${crypto.symbol}`}
                          className="btn btn-outline-primary btn-sm"
                          title="График"
                        >
                          <i className="bi bi-graph-up"></i>
                        </a>
                        <button
                          className="btn btn-outline-secondary btn-sm"
                          onClick={() => onAddToWatchlist(crypto.symbol)}
                          title="В избранное"
                        >
                          <i className="bi bi-star"></i>
                        </button>
                        <button
                          className="btn btn-outline-warning btn-sm"
                          onClick={() => onSetAlert(crypto)}
                          title="Алерт"
                        >
                          <i className="bi bi-bell"></i>
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card-footer bg-white border-0">
        <div className="d-flex justify-content-between align-items-center">
          <div className="text-muted small">
            Показано {paginatedData.length} из {sortedData.length} записей
            {sortedData.length !== data.length && ` (отфильтровано из ${data.length})`}
          </div>
          {totalPages > 1 && renderPagination()}
        </div>
      </div>
    </div>
  );
};

export default MarketTable;
