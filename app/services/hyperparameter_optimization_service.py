"""
Сервис оптимизации гиперпараметров для криптовалютных ML моделей
Использует GridSearchCV и RandomizedSearchCV для поиска оптимальных параметров
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from flask import current_app
import joblib
import time
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None


class HyperparameterOptimizationService:
    """
    Сервис для оптимизации гиперпараметров ML моделей
    """
    
    def __init__(self):
        self.best_params = {}
        self.optimization_results = {}
        
    def get_parameter_grids(self) -> Dict[str, Dict[str, Any]]:
        """
        Получить сетки параметров для оптимизации
        """
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'gb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if XGB_AVAILABLE:
            param_grids['xgb'] = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            }
            
        return param_grids
    
    def get_reduced_parameter_grids(self) -> Dict[str, Dict[str, Any]]:
        """
        Получить уменьшенные сетки параметров для быстрой оптимизации
        """
        param_grids = {
            'rf': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', None]
            },
            'gb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
        }
        
        if XGB_AVAILABLE:
            param_grids['xgb'] = {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
        return param_grids
    
    def optimize_model_hyperparameters(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        use_reduced_grid: bool = True,
        cv_folds: int = 3,
        n_jobs: int = -1,
        scoring: str = 'neg_mean_absolute_error'
    ) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
        """
        Оптимизировать гиперпараметры для конкретной модели
        """
        current_app.logger.info(f"🔧 Начинаем оптимизацию гиперпараметров для {model_name}")
        start_time = time.time()
        
        # Получаем базовую модель
        base_model = self._get_base_model(model_name)
        if base_model is None:
            raise ValueError(f"Неподдерживаемая модель: {model_name}")
        
        # Получаем сетку параметров
        param_grids = self.get_reduced_parameter_grids() if use_reduced_grid else self.get_parameter_grids()
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            current_app.logger.warning(f"Нет параметров для оптимизации модели {model_name}")
            return base_model, {}, {}
        
        # Настраиваем кросс-валидацию для временных рядов
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Выполняем GridSearch
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0,
            error_score='raise'
        )
        
        try:
            # Очищаем данные
            X_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
            y_clean = y_train.fillna(y_train.mean())
            
            # Выполняем поиск
            grid_search.fit(X_clean, y_clean)
            
            # Получаем результаты
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Конвертируем обратно из negative MAE
            
            # Вычисляем дополнительные метрики
            y_pred = best_model.predict(X_clean)
            r2 = r2_score(y_clean, y_pred)
            mae = mean_absolute_error(y_clean, y_pred)
            
            metrics = {
                'best_cv_mae': best_score,
                'train_mae': mae,
                'train_r2': r2,
                'optimization_time': time.time() - start_time
            }
            
            # Сохраняем результаты
            self.best_params[model_name] = best_params
            self.optimization_results[model_name] = {
                'best_params': best_params,
                'metrics': metrics,
                'cv_results': grid_search.cv_results_
            }
            
            current_app.logger.info(
                f"✅ Оптимизация {model_name} завершена за {metrics['optimization_time']:.2f}с. "
                f"Лучший CV MAE: {best_score:.2f}, R²: {r2:.3f}"
            )
            
            return best_model, best_params, metrics
            
        except Exception as e:
            current_app.logger.error(f"❌ Ошибка при оптимизации {model_name}: {e}")
            return base_model, {}, {'error': str(e)}
    
    def optimize_ensemble_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models_to_optimize: list = None,
        use_reduced_grid: bool = True,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Оптимизировать гиперпараметры для всех моделей в ансамбле
        """
        if models_to_optimize is None:
            models_to_optimize = ['rf', 'gb']
            if XGB_AVAILABLE:
                models_to_optimize.append('xgb')
        
        current_app.logger.info(f"🚀 Начинаем оптимизацию ансамбля моделей: {models_to_optimize}")
        
        optimized_models = {}
        optimization_summary = {}
        
        for model_name in models_to_optimize:
            try:
                model, params, metrics = self.optimize_model_hyperparameters(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    use_reduced_grid=use_reduced_grid,
                    cv_folds=cv_folds
                )
                
                optimized_models[model_name] = model
                optimization_summary[model_name] = {
                    'best_params': params,
                    'metrics': metrics
                }
                
            except Exception as e:
                current_app.logger.error(f"❌ Не удалось оптимизировать {model_name}: {e}")
                # Используем базовую модель в случае ошибки
                base_model = self._get_base_model(model_name)
                if base_model:
                    optimized_models[model_name] = base_model
                    optimization_summary[model_name] = {
                        'best_params': {},
                        'metrics': {'error': str(e)}
                    }
        
        return {
            'optimized_models': optimized_models,
            'optimization_summary': optimization_summary,
            'total_models_optimized': len(optimized_models)
        }
    
    def _get_base_model(self, model_name: str) -> Optional[Any]:
        """
        Получить базовую модель по имени
        """
        if model_name == 'rf':
            return RandomForestRegressor(random_state=42, n_jobs=-1)
        elif model_name == 'gb':
            return GradientBoostingRegressor(random_state=42)
        elif model_name == 'xgb' and XGB_AVAILABLE:
            return xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                objective='reg:squarederror'
            )
        else:
            return None
    
    def save_optimization_results(self, filepath: str):
        """
        Сохранить результаты оптимизации
        """
        try:
            results = {
                'best_params': self.best_params,
                'optimization_results': self.optimization_results
            }
            joblib.dump(results, filepath)
            current_app.logger.info(f"💾 Результаты оптимизации сохранены в {filepath}")
        except Exception as e:
            current_app.logger.error(f"❌ Ошибка при сохранении результатов оптимизации: {e}")
    
    def load_optimization_results(self, filepath: str) -> bool:
        """
        Загрузить результаты оптимизации
        """
        try:
            results = joblib.load(filepath)
            self.best_params = results.get('best_params', {})
            self.optimization_results = results.get('optimization_results', {})
            current_app.logger.info(f"📂 Результаты оптимизации загружены из {filepath}")
            return True
        except Exception as e:
            current_app.logger.warning(f"⚠️ Не удалось загрузить результаты оптимизации: {e}")
            return False
    
    def get_optimization_report(self) -> str:
        """
        Получить отчет об оптимизации
        """
        if not self.optimization_results:
            return "Оптимизация еще не выполнялась"
        
        report = "🔧 ОТЧЕТ ОБ ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ\n"
        report += "=" * 60 + "\n\n"
        
        for model_name, results in self.optimization_results.items():
            report += f"📊 Модель: {model_name.upper()}\n"
            report += "-" * 30 + "\n"
            
            best_params = results.get('best_params', {})
            metrics = results.get('metrics', {})
            
            if best_params:
                report += "Лучшие параметры:\n"
                for param, value in best_params.items():
                    report += f"  {param}: {value}\n"
                report += "\n"
            
            if metrics and 'error' not in metrics:
                report += "Метрики производительности:\n"
                report += f"  CV MAE: {metrics.get('best_cv_mae', 'N/A'):.2f}\n"
                report += f"  Train MAE: {metrics.get('train_mae', 'N/A'):.2f}\n"
                report += f"  Train R²: {metrics.get('train_r2', 'N/A'):.3f}\n"
                report += f"  Время оптимизации: {metrics.get('optimization_time', 'N/A'):.2f}с\n"
            elif 'error' in metrics:
                report += f"❌ Ошибка: {metrics['error']}\n"
            
            report += "\n"
        
        return report
