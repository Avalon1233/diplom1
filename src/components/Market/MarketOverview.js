import React, { useMemo } from 'react';

const MarketOverview = ({ data, loading }) => {
  const marketStats = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        totalPairs: 0,
        risingCount: 0,
        fallingCount: 0,
        avgVolatility: 0,
        totalVolume: 0
      };
    }

    const risingCount = data.filter(crypto => crypto.change >= 0).length;
    const fallingCount = data.length - risingCount;
    const avgVolatility = data.reduce((sum, crypto) => sum + Math.abs(crypto.change), 0) / data.length;
    const totalVolume = data.reduce((sum, crypto) => sum + crypto.volume, 0);

    return {
      totalPairs: data.length,
      risingCount,
      fallingCount,
      avgVolatility,
      totalVolume
    };
  }, [data]);

  const formatNumber = (num) => {
    return new Intl.NumberFormat('en-US').format(Math.round(num));
  };

  const formatVolume = (volume) => {
    if (volume >= 1e9) {
      return `${(volume / 1e9).toFixed(1)}B`;
    } else if (volume >= 1e6) {
      return `${(volume / 1e6).toFixed(1)}M`;
    } else if (volume >= 1e3) {
      return `${(volume / 1e3).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const StatCard = ({ title, value, subtitle, icon, color, trend }) => (
    <div className="col-xl-3 col-md-6 mb-4">
      <div className={`card border-left-${color} shadow h-100 py-2`}>
        <div className="card-body">
          <div className="row no-gutters align-items-center">
            <div className="col mr-2">
              <div className={`text-xs font-weight-bold text-${color} text-uppercase mb-1`}>
                {title}
              </div>
              <div className="h5 mb-0 font-weight-bold text-gray-800">
                {loading ? (
                  <div className="loading-placeholder">
                    <div className="loading-spinner"></div>
                  </div>
                ) : (
                  value
                )}
              </div>
              {subtitle && (
                <div className="mt-2">
                  <small className="text-muted">{subtitle}</small>
                </div>
              )}
            </div>
            <div className="col-auto">
              <i className={`bi ${icon} fa-2x text-gray-300`}></i>
            </div>
          </div>
          {trend && (
            <div className="row no-gutters align-items-center mt-2">
              <div className="col-auto">
                <i className={`bi ${trend.direction === 'up' ? 'bi-arrow-up text-success' : 'bi-arrow-down text-danger'} me-1`}></i>
              </div>
              <div className="col">
                <div className="progress progress-sm mr-2">
                  <div 
                    className={`progress-bar ${trend.direction === 'up' ? 'bg-success' : 'bg-danger'}`}
                    style={{ width: `${Math.min(trend.percentage, 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="row mb-4">
      <StatCard
        title="Всего пар"
        value={formatNumber(marketStats.totalPairs)}
        subtitle="Криптовалютных пар"
        icon="bi-collection"
        color="primary"
      />
      
      <StatCard
        title="Растущие"
        value={formatNumber(marketStats.risingCount)}
        subtitle={`${marketStats.totalPairs > 0 ? ((marketStats.risingCount / marketStats.totalPairs) * 100).toFixed(1) : 0}% от общего числа`}
        icon="bi-arrow-up-circle"
        color="success"
        trend={{
          direction: 'up',
          percentage: marketStats.totalPairs > 0 ? (marketStats.risingCount / marketStats.totalPairs) * 100 : 0
        }}
      />
      
      <StatCard
        title="Падающие"
        value={formatNumber(marketStats.fallingCount)}
        subtitle={`${marketStats.totalPairs > 0 ? ((marketStats.fallingCount / marketStats.totalPairs) * 100).toFixed(1) : 0}% от общего числа`}
        icon="bi-arrow-down-circle"
        color="danger"
        trend={{
          direction: 'down',
          percentage: marketStats.totalPairs > 0 ? (marketStats.fallingCount / marketStats.totalPairs) * 100 : 0
        }}
      />
      
      <StatCard
        title="Общий объём"
        value={`$${formatVolume(marketStats.totalVolume)}`}
        subtitle={`Средняя волатильность: ${marketStats.avgVolatility.toFixed(2)}%`}
        icon="bi-bar-chart"
        color="info"
      />
    </div>
  );
};

export default MarketOverview;
