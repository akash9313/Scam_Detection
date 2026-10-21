"""
Ablation and perturbation analysis on UPI and Jio failing fixtures.
Tests various sub-components to identify the exact lexical/syntactic triggers for high spam scores.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import predict_message

def test_ablations(method: str):
    root = str(PROJECT_ROOT)
    
    upi_parts = [
        ("Full UPI fixture", "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"),
        ("UPI Part 1 only (Dr./Cr.)", "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi."),
        ("UPI Part 1 + Ref", "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737."),
        ("UPI Part 1 + Ref + AvlBal", "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20)"),
        ("UPI without Call clause", "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79"),
        ("Only Call clause", "Not you? Call 18005700/5/5000-BOB"),
        ("Call without BOB", "Not you? Call 18005700"),
        ("Expanded Dr/Cr to debited/credited", "Rs. 1000.00 debited from A/C XXXXXXXXX3220 and credited to 9313357315@upi. Ref: 075319983737. Available balance: Rs 175527.79. Not you? Call 18005700/5/5000-BOB"),
        ("Full verbal banking", "Rs. 1000.00 debited from Account 3220 and credited to user@upi. Ref 075319983737. Available Balance Rs 175527.79."),
    ]

    print(f"\n=================== UPI ABLATIONS ({method}) ===================")
    for name, p in upi_parts:
        res = predict_message(p, method, root)
        print(f"{res['spam_probability']*100:6.2f}% SPAM | {res['decision']:15} | [{name}]: {p[:65]}...")

    jio_parts = [
        ("Full Jio fixture", "Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"),
        ("Sentence 1 only", "Reverification process of your Jio Number 9313357315 has been initiated."),
        ("Sentence 1 + 2", "Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed."),
        ("Disclaimer clause only", "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation."),
        ("Helpline clause only", "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"),
        ("Call number only", "please call us on 18008899999"),
        ("Without Helpline", "Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation."),
        ("Without Disclaimer", "Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"),
        ("Synthetic telecom fixture from Phase 3", "Dear Customer, your mobile number reverification request has been successfully submitted. You will receive an SMS confirmation once the verification process is complete. For customer care assistance, please call 198 or +91-98765XXXXX or email care@telecom-service.com. Thank you for staying with us."),
    ]

    print(f"\n=================== JIO ABLATIONS ({method}) ===================")
    for name, p in jio_parts:
        res = predict_message(p, method, root)
        print(f"{res['spam_probability']*100:6.2f}% SPAM | {res['decision']:15} | [{name}]: {p[:65]}...")

if __name__ == "__main__":
    for m in ["lora_v2", "full_finetuning_v2", "lora"]:
        test_ablations(m)
