"""Comprehensive test of all modules — simulates actual data flow."""
import sys
import pandas as pd
import numpy as np
import torch

n = 200
np.random.seed(42)
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
df = pd.DataFrame({
    'open': close * 0.999, 'high': close * 1.01, 'low': close * 0.99,
    'close': close, 'volume': np.random.randint(1000, 10000, n),
    'trade_time': pd.date_range('2025-05-14 09:30', periods=n, freq='5min')
})

errors = []

# 1. Features
try:
    from data.features import compute_all_indicators
    ind = compute_all_indicators(df)
    # Merge into df (simulating real data flow)
    for col in ind.columns:
        if col not in df.columns:
            df[col] = ind[col].values
    print(f'[OK] Features: ind={ind.shape}, merged_df={df.shape}')
except Exception as e: print(f'[FAIL] Features: {e}'); errors.append(e)

# 2. Preprocess (on merged df)
try:
    from data.preprocessor import preprocess, build_targets, FEATURE_COLS
    arr, sc = preprocess(df, fit_scaler=False)  # predictor has its own scaler
    dirs, prices = build_targets(df['close'].values, horizon=10)
    print(f'[OK] Preprocess: arr={arr.shape}, pool={len(FEATURE_COLS)}')
except Exception as e: print(f'[FAIL] Preprocess: {e}'); errors.append(e)

# 3. Counter-prediction
try:
    from data.counter_prediction import counter_trade_features
    cpf = counter_trade_features(df)
    print(f'[OK] Counter-prediction: {cpf.shape}')
except Exception as e: print(f'[FAIL] Counter-prediction: {e}'); errors.append(e)

# 4. Trade advice
try:
    from data.trade_advice import generate_trade_advice
    pred_in = {'direction': 'up', 'direction_conf': 0.75, 'target_price': close[-1] * 1.01}
    advice = generate_trade_advice(pred_in, df)
    print(f'[OK] Trade advice: action={advice["action"]}, pct={advice["position_pct"]}%')
except Exception as e: print(f'[FAIL] Trade advice: {e}'); errors.append(e)

# 5. News analyzer
try:
    from data.news_analyzer import analyze_news_impacts, has_news_impact, compute_news_bias
    news = [{'title': '某某公司业绩大幅增长超预期签订重大合同', 'content': '', 'time': '2025-05-14 10:00:00'}]
    has = has_news_impact(news)
    print(f'[OK] News impact: {has}')
    impacts = analyze_news_impacts(news)
    bias = compute_news_bias(impacts, pd.Timestamp.now())
    print(f'[OK] News bias: bias={bias["bias"]}, active={bias["active_count"]}')
except Exception as e: print(f'[FAIL] News: {e}'); errors.append(e)

# 6. Feature selector (on merged df with indicators + OHLCV)
try:
    from data.feature_selector import select_features_mrmr, get_feature_mask, FEATURE_POOL
    selected = select_features_mrmr(df, k=40, horizon=10)
    print(f'[OK] MRMR: selected {len(selected)}/{len(FEATURE_POOL)} features')
    mask = get_feature_mask(selected)
    masked_arr = arr[:, mask]
    from sklearn.preprocessing import StandardScaler
    sc2 = StandardScaler(); sc2.fit(masked_arr)
    print(f'[OK] Feature mask: {arr.shape} -> {masked_arr.shape}')
except Exception as e: print(f'[FAIL] Feature selector: {e}'); errors.append(e)

# 7. Dataset
try:
    from model.dataset import StockDataset
    seqs = [masked_arr]; tgts = [np.stack([dirs, prices], axis=1)]
    ds = StockDataset(seqs, tgts, seq_len=120, horizon=10)
    x, d, p = ds[0]
    print(f'[OK] Dataset: {len(ds)} samples, x={x.shape}')
except Exception as e: print(f'[FAIL] Dataset: {e}'); errors.append(e)

# 8. Model with selected dim (matching config defaults)
try:
    from model.lstm_transformer import HybridModel
    from config import AppConfig
    _cfg = AppConfig()
    input_dim = len(selected)
    model = HybridModel(input_dim=input_dim, d_model=_cfg.model.d_model,
                        lstm_hidden=_cfg.model.lstm_hidden,
                        lstm_layers=_cfg.model.lstm_layers,
                        transformer_layers=_cfg.model.transformer_layers,
                        nhead=_cfg.model.nhead, dropout=_cfg.model.dropout,
                        max_seq_len=240, patch_len=_cfg.model.patch_len)
    params = sum(p.numel() for p in model.parameters())
    x_in = torch.randn(2, 240, input_dim)
    dir_out, price_out = model(x_in)
    print(f'[OK] Model: {params:,} params, dir={dir_out.shape}, price={price_out.shape}')
except Exception as e: print(f'[FAIL] Model: {e}'); errors.append(e)

# 9. GUI imports
try:
    from gui.chart_canvas import ChartCanvas
    from gui.prediction_dashboard import PredictionDashboard
    print('[OK] GUI chart + dashboard')
except Exception as e: print(f'[FAIL] GUI: {e}'); errors.append(e)

# 10. MainWindow import
try:
    from gui.main_window import MainWindow
    print('[OK] MainWindow')
except Exception as e: print(f'[FAIL] MainWindow: {e}'); errors.append(e)

# 11. Dialogs import
try:
    from gui.dialogs import TrainDialog, SettingsDialog
    print('[OK] Dialogs')
except Exception as e: print(f'[FAIL] Dialogs: {e}'); errors.append(e)

# 12. Predictor (full cycle with checkpoint)
pred_r = None  # in case of early failure, test 13 can skip gracefully
try:
    from model.predictor import Predictor
    from config import AppConfig
    import os, tempfile
    cfg = AppConfig()
    cfg.model.feature_dim = len(FEATURE_POOL)
    cfg.model.seq_len = 240  # match test #8 model's max_seq_len (patchTST)
    tmp_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmp_dir, 'test_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': sc2,
        'selected_features': selected,
        'model_version': 'test',
        'epoch': 0, 'val_loss': 0.5, 'val_acc': 0.6
    }, ckpt_path)
    pred_r = Predictor(ckpt_path, cfg)
    result = pred_r.predict_one(arr, close[-1])
    print(f'[OK] Predictor: {result["direction"]}, target={result["target_price"]}')
    os.remove(ckpt_path); os.rmdir(tmp_dir)
except Exception as e: print(f'[FAIL] Predictor: {e}'); errors.append(e)

# 13. Online correction
try:
    if pred_r is None:
        raise RuntimeError("Predictor not initialized (test 12 failed)")
    correction = pred_r.online_correct(arr, result["target_price"], close[-1] * 1.002)
    print(f'[OK] Correction: corrected={correction["corrected"]}, error={correction.get("error_pct", 0):.4%}')
except Exception as e: print(f'[FAIL] Correction: {e}'); errors.append(e)

# 14. Trade advice edge cases
try:
    # Test flat direction
    adv_flat = generate_trade_advice({'direction': 'flat', 'direction_conf': 0.3}, df)
    assert adv_flat['action'] == '观望'
    # Test empty df
    adv_empty = generate_trade_advice({'direction': 'up'}, pd.DataFrame())
    assert adv_empty['action'] == '数据不足'
    print('[OK] Trade advice edge cases')
except Exception as e: print(f'[FAIL] Trade edge cases: {e}'); errors.append(e)

print()
if errors:
    print(f'FAILED {len(errors)} test(s):')
    for e in errors: print(f'  - {e}')
else:
    print('ALL 14 TESTS PASSED')
