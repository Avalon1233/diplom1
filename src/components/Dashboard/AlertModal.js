import React, { useState, useEffect } from 'react';

const AlertModal = ({ crypto, onClose, onCreateAlert }) => {
  const [condition, setCondition] = useState('>');
  const [targetPrice, setTargetPrice] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Set initial target price based on current price
    if (crypto && crypto.price) {
      const suggestedPrice = condition === '>' 
        ? (crypto.price * 1.05).toFixed(2)
        : (crypto.price * 0.95).toFixed(2);
      setTargetPrice(suggestedPrice);
    }
  }, [crypto, condition]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!targetPrice || isNaN(targetPrice)) {
      alert('Пожалуйста, укажите корректную целевую цену');
      return;
    }

    setLoading(true);
    
    try {
      await onCreateAlert({
        condition,
        target_price: parseFloat(targetPrice)
      });
    } catch (error) {
      console.error('Error creating alert:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: price < 1 ? 4 : 2
    }).format(price);
  };

  return (
    <div className="modal fade show alert-modal" style={{ display: 'block' }} tabIndex="-1">
      <div className="modal-backdrop fade show" onClick={onClose}></div>
      <div className="modal-dialog">
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">
              <i className="bi bi-bell me-2"></i>
              Установить ценовой алерт
            </h5>
            <button 
              type="button" 
              className="btn-close" 
              onClick={onClose}
              disabled={loading}
            ></button>
          </div>
          
          <form onSubmit={handleSubmit}>
            <div className="modal-body">
              <div className="mb-3">
                <label className="form-label">Криптовалюта</label>
                <div className="input-group">
                  <span className="input-group-text">
                    <i className="bi bi-coin"></i>
                  </span>
                  <input
                    type="text"
                    className="form-control"
                    value={crypto.symbol}
                    readOnly
                  />
                </div>
                <div className="form-text">
                  Текущая цена: <strong>{formatPrice(crypto.price)}</strong>
                </div>
              </div>

              <div className="mb-3">
                <label className="form-label">Условие</label>
                <select
                  className="form-select"
                  value={condition}
                  onChange={(e) => setCondition(e.target.value)}
                  disabled={loading}
                >
                  <option value=">">Цена выше</option>
                  <option value="<">Цена ниже</option>
                </select>
              </div>

              <div className="mb-3">
                <label className="form-label">Целевая цена ($)</label>
                <div className="input-group">
                  <span className="input-group-text">$</span>
                  <input
                    type="number"
                    className="form-control"
                    value={targetPrice}
                    onChange={(e) => setTargetPrice(e.target.value)}
                    step="0.01"
                    min="0"
                    required
                    disabled={loading}
                    placeholder="Введите целевую цену"
                  />
                </div>
                <div className="form-text">
                  {condition === '>' ? 'Алерт сработает когда цена поднимется выше' : 'Алерт сработает когда цена опустится ниже'} ${targetPrice || '0.00'}
                </div>
              </div>

              {/* Price suggestion buttons */}
              <div className="mb-3">
                <label className="form-label">Быстрый выбор:</label>
                <div className="btn-group w-100" role="group">
                  <button
                    type="button"
                    className="btn btn-outline-secondary btn-sm"
                    onClick={() => setTargetPrice((crypto.price * 0.95).toFixed(2))}
                    disabled={loading}
                  >
                    -5%
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary btn-sm"
                    onClick={() => setTargetPrice((crypto.price * 0.90).toFixed(2))}
                    disabled={loading}
                  >
                    -10%
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary btn-sm"
                    onClick={() => setTargetPrice(crypto.price.toFixed(2))}
                    disabled={loading}
                  >
                    Текущая
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary btn-sm"
                    onClick={() => setTargetPrice((crypto.price * 1.05).toFixed(2))}
                    disabled={loading}
                  >
                    +5%
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary btn-sm"
                    onClick={() => setTargetPrice((crypto.price * 1.10).toFixed(2))}
                    disabled={loading}
                  >
                    +10%
                  </button>
                </div>
              </div>
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={onClose}
                disabled={loading}
              >
                Отмена
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !targetPrice}
              >
                {loading ? (
                  <>
                    <span className="loading-spinner me-2"></span>
                    Создание...
                  </>
                ) : (
                  <>
                    <i className="bi bi-bell me-2"></i>
                    Создать алерт
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AlertModal;
