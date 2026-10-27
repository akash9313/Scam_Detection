import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from app.inference import predict_message

bob_msg = "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
jio_msg = "Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
spam_msg = "Dear customer your SBI YONO account will be suspended today. Please update your PAN immediately by visiting http://sbi-kyc-update.top/pan to avoid blockage."

print("======================================================================")
print("REHEARSAL OF 4 DEMO FIXTURES ON LORA_V3 (PRODUCTION DEFAULT)")
print("======================================================================")

res_bob = predict_message(bob_msg, "lora_v3", str(root))
print("\n--- FIXTURE 1: BANK OF BARODA UPI TRANSACTION ---")
print("Input:", bob_msg)
print("Predicted Label :", res_bob["predicted_label"])
print("Decision Banner :", res_bob["decision"])
print("P(SPAM)         :", f"{res_bob['spam_probability']*100:.2f}%")
print("P(HAM)          :", f"{res_bob['ham_probability']*100:.2f}%")
print("Threshold (tau) :", res_bob["threshold_used"])
print("Abstention Band :", f"[{res_bob['tau_low']}, {res_bob['tau_high']})")

res_jio = predict_message(jio_msg, "lora_v3", str(root))
print("\n--- FIXTURE 2: RELIANCE JIO REVERIFICATION NOTIFICATION ---")
print("Input:", jio_msg)
print("Predicted Label :", res_jio["predicted_label"])
print("Decision Banner :", res_jio["decision"])
print("P(SPAM)         :", f"{res_jio['spam_probability']*100:.2f}%")
print("P(HAM)          :", f"{res_jio['ham_probability']*100:.2f}%")
print("Threshold (tau) :", res_jio["threshold_used"])
print("Abstention Band :", f"[{res_jio['tau_low']}, {res_jio['tau_high']})")

res_spam = predict_message(spam_msg, "lora_v3", str(root))
print("\n--- FIXTURE 3: CLEAR SPAM / PHISHING NOTIFICATION ---")
print("Input:", spam_msg)
print("Predicted Label :", res_spam["predicted_label"])
print("Decision Banner :", res_spam["decision"])
print("P(SPAM)         :", f"{res_spam['spam_probability']*100:.2f}%")
print("P(HAM)          :", f"{res_spam['ham_probability']*100:.2f}%")
print("Threshold (tau) :", res_spam["threshold_used"])
print("Abstention Band :", f"[{res_spam['tau_low']}, {res_spam['tau_high']})")

candidates = [
    "Exclusive weekend offer: Get 50% cashback on your next transaction at partner stores.",
    "Alert: Login attempt from new device in Mumbai. If not you, contact support.",
    "Your monthly subscription renewal is pending. Click here to confirm payment details.",
    "Reminder: your appointment is scheduled for tomorrow at 4 PM. Reply YES to confirm or NO to cancel.",
    "Dear subscriber, special discount offer on recharge of 299. Valid today only.",
    "Claim your free points now by clicking http://bit.ly/points-claim before midnight.",
    "Urgent: We attempted to deliver your parcel today. Schedule redelivery at http://delivery-post.info",
    "Dear valued customer, your policy number 49201 is due for annual renewal. Call toll free 1800-222-111 for discount options.",
    "Hi Akash, please find the attached receipt for your order #8921. For queries visit our helpdesk.",
    "Your Netflix membership payment was declined. Update card to continue streaming: http://netflix-billing-update.com",
    "Flat 500 off on your next flight booking with Code FLY500. Book now on app.",
    "Dear customer, you have unused reward points worth Rs 450 expiring today. Redeem at reward-store-online.in",
]

print("\n--- TESTING BORDERLINE CANDIDATES FOR REVIEW_REQUIRED ([0.25, 0.75)) ---")
found_borderline = None
for c in candidates:
    r = predict_message(c, "lora_v3", str(root))
    p_spam_pct = r["spam_probability"] * 100
    p_ham_pct = r["ham_probability"] * 100
    print(f"[{r['decision']:15}] P(SPAM)={p_spam_pct:6.2f}% | P(HAM)={p_ham_pct:6.2f}% | Text: {c[:60]}...")
    if r["decision"] == "REVIEW_REQUIRED" and found_borderline is None:
        found_borderline = (c, r)

if found_borderline:
    msg, r = found_borderline
    print("\n--- FIXTURE 4: VERIFIED BORDERLINE (REVIEW_REQUIRED) ---")
    print("Input:", msg)
    print("Predicted Label :", r["predicted_label"])
    print("Decision Banner :", r["decision"])
    print("P(SPAM)         :", f"{r['spam_probability']*100:.2f}%")
    print("P(HAM)          :", f"{r['ham_probability']*100:.2f}%")
    print("Threshold (tau) :", r["threshold_used"])
    print("Abstention Band :", f"[{r['tau_low']}, {r['tau_high']})")
else:
    print("\nNo candidate fell into [0.25, 0.75).")
