# Plain-English alert generator — no API needed, pure rule-based NLG:
def generate_alert(row: dict, shap_top: list) -> str:
    reasons = []
    feature_map = {
        'amount':              f"Amount ₹{row.get('amount',0):,.0f} is unusually high",
        'amount_to_avg_ratio': f"Amount is {row.get('amount_to_avg_ratio',1):.1f}x your 7-day average",
        'hour':                f"Transaction at {int(row.get('hour',12))}:00 (odd hour)",
        'is_new_merchant':     "Payment to a new/unknown merchant",
        'txn_per_day':         f"{int(row.get('txn_per_day',1))} transactions today (unusually high)",
        'device_change':       "Transaction from an unrecognized device",
        'location_change':     "Location differs from your usual area",
        'merchant_cat_enc':    f"Unusual merchant category for you",
    }
    for feat, impact in shap_top:
        if impact > 0 and feat in feature_map:
            reasons.append(feature_map[feat])

    prob = row.get('fraud_probability', 0)
    if prob >= 75:
        severity = "HIGH RISK"
        emoji = "🚨"
    elif prob >= 45:
        severity = "MEDIUM RISK"
        emoji = "⚠️"
    else:
        severity = "LOW RISK"
        emoji = "🔔"

    merchant = row.get('merchant_cat', 'unknown')
    amount = row.get('amount', 0)

    alert = f"{emoji} {severity} — Suspicious transaction detected\n\n"
    alert += f"₹{amount:,.0f} at {merchant} merchant\n"
    alert += f"Fraud probability: {prob:.1f}%\n\n"
    if reasons:
        alert += "Reasons flagged:\n"
        for r in reasons:
            alert += f"  • {r}\n"
    alert += "\nWas this you? If not, contact your bank immediately."
    return alert