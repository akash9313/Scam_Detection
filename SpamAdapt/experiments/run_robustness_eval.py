"""
SpamAdapt Phase 5: Robustness Testing & Evaluation Suite.

Constructs controlled perturbation fixtures (data/robustness/robustness_fixtures.csv)
and evaluates prediction consistency, score stability, and class flips across
frozen models: Full Fine-Tuning v2, LoRA v2, and DoRA v2.

Perturbation Types:
1. casing_all_caps: Conversion to UPPERCASE
2. casing_all_lower: Conversion to lowercase
3. whitespace_punctuation: Extra spacing, repeated punctuation (!!, ..., ??)
4. benign_paraphrase: Intent-preserving synonymous rephrasings
5. redaction_markers: Replacement of numbers and links with [PHONE], [LINK], [NUMBER]
6. urgency_perturbation: Intent-preserving urgency softening/amplification
7. code_switching_equivalence: Parallel English <-> Hinglish rephrasings

Outputs:
- data/robustness/robustness_fixtures.csv
- results/phase5/robustness_metrics.json
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import predict_message, get_default_threshold_for_method

ROBUSTNESS_FIXTURES = [
    # -------------------------------------------------------------
    # BASE 1: Telecom Reverification (HAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_TEL_01",
        "perturbation_id": "ROB_TEL_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Dear Customer, your mobile number reverification request has been successfully submitted. You will receive an SMS confirmation once completed. For support call 198 or email care@telecom-service.com.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "telecom_service",
        "description": "Original verified telecom service notice baseline"
    },
    {
        "base_id": "BASE_TEL_01",
        "perturbation_id": "ROB_TEL_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "DEAR CUSTOMER, YOUR MOBILE NUMBER REVERIFICATION REQUEST HAS BEEN SUCCESSFULLY SUBMITTED. YOU WILL RECEIVE AN SMS CONFIRMATION ONCE COMPLETED. FOR SUPPORT CALL 198 OR EMAIL CARE@TELECOM-SERVICE.COM.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "telecom_service",
        "description": "Full uppercase casing perturbation"
    },
    {
        "base_id": "BASE_TEL_01",
        "perturbation_id": "ROB_TEL_01_LOWER",
        "perturbation_type": "casing_all_lower",
        "sms": "dear customer, your mobile number reverification request has been successfully submitted. you will receive an sms confirmation once completed. for support call 198 or email care@telecom-service.com.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "telecom_service",
        "description": "Full lowercase casing perturbation"
    },
    {
        "base_id": "BASE_TEL_01",
        "perturbation_id": "ROB_TEL_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "Dear Customer ,  your mobile number reverification request has been successfully submitted !! You will receive an SMS confirmation once completed ... For support call 198 or email care@telecom-service.com .",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "telecom_service",
        "description": "Whitespace padding and repeated punctuation"
    },
    {
        "base_id": "BASE_TEL_01",
        "perturbation_id": "ROB_TEL_01_PARA",
        "perturbation_type": "benign_paraphrase",
        "sms": "Hello subscriber, your mobile number verification submission is under review. An SMS confirmation will be sent upon completion. For queries, contact helpline 198 or care@telecom-service.com.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "telecom_service",
        "description": "Synonymous benign paraphrase preserving formal service tone"
    },
    {
        "base_id": "BASE_TEL_01",
        "perturbation_id": "ROB_TEL_01_REDACT",
        "perturbation_type": "redaction_markers",
        "sms": "Dear Customer, your mobile number reverification request has been successfully submitted. You will receive an SMS confirmation once completed. For support call [PHONE] or email [EMAIL].",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "telecom_service",
        "description": "Explicit entity redaction markers"
    },

    # -------------------------------------------------------------
    # BASE 2: Bank OTP Notification (HAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_OTP_01",
        "perturbation_id": "ROB_OTP_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Your OTP for login to NetBanking is 492018. Valid for 5 minutes. Do not share this OTP with anyone, including bank officials.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "bank_otp",
        "description": "Standard banking authentication OTP baseline"
    },
    {
        "base_id": "BASE_OTP_01",
        "perturbation_id": "ROB_OTP_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "YOUR OTP FOR LOGIN TO NETBANKING IS 492018. VALID FOR 5 MINUTES. DO NOT SHARE THIS OTP WITH ANYONE, INCLUDING BANK OFFICIALS.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "bank_otp",
        "description": "All caps OTP message"
    },
    {
        "base_id": "BASE_OTP_01",
        "perturbation_id": "ROB_OTP_01_LOWER",
        "perturbation_type": "casing_all_lower",
        "sms": "your otp for login to netbanking is 492018. valid for 5 minutes. do not share this otp with anyone, including bank officials.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "bank_otp",
        "description": "All lowercase OTP message"
    },
    {
        "base_id": "BASE_OTP_01",
        "perturbation_id": "ROB_OTP_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "Your OTP for login to NetBanking is 492018 !!! Valid for 5 minutes ... Do NOT share this OTP with anyone , including bank officials !!",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "bank_otp",
        "description": "Exclamation emphasis and spaced punctuation"
    },
    {
        "base_id": "BASE_OTP_01",
        "perturbation_id": "ROB_OTP_01_PARA",
        "perturbation_type": "benign_paraphrase",
        "sms": "Use one-time passcode 492018 to access your NetBanking account. Expires in 5 minutes. Never disclose this security code to any person.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "bank_otp",
        "description": "Alternative security wording preserving authentic advisory"
    },
    {
        "base_id": "BASE_OTP_01",
        "perturbation_id": "ROB_OTP_01_URG",
        "perturbation_type": "urgency_perturbation",
        "sms": "Your OTP for login to NetBanking is 492018. Please use it within a short period. Do not share this OTP with anyone, including bank officials.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "bank_otp",
        "description": "Softened time window phrasing without numerical urgency"
    },

    # -------------------------------------------------------------
    # BASE 3: UPI Transaction Alert (HAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_UPI_01",
        "perturbation_id": "ROB_UPI_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Rs. 450.00 debited from A/c XX4821 on 14-Oct-26 via UPI Ref 428901827361 to Merchant Store. Avail Bal: Rs. 12,450.00.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "upi_transaction",
        "description": "Standard UPI debit alert baseline"
    },
    {
        "base_id": "BASE_UPI_01",
        "perturbation_id": "ROB_UPI_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "RS. 450.00 DEBITED FROM A/C XX4821 ON 14-OCT-26 VIA UPI REF 428901827361 TO MERCHANT STORE. AVAIL BAL: RS. 12,450.00.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "upi_transaction",
        "description": "All uppercase debit notification"
    },
    {
        "base_id": "BASE_UPI_01",
        "perturbation_id": "ROB_UPI_01_PARA",
        "perturbation_type": "benign_paraphrase",
        "sms": "INR 450.00 was transferred from account ending in 4821 on 14 Oct 2026 using UPI reference 428901827361 to Merchant Store. Current balance is INR 12,450.00.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "upi_transaction",
        "description": "Conversational banking phrasing"
    },
    {
        "base_id": "BASE_UPI_01",
        "perturbation_id": "ROB_UPI_01_REDACT",
        "perturbation_type": "redaction_markers",
        "sms": "[AMOUNT] debited from A/c [ACCOUNT] on [DATE] via UPI Ref [REF_NO] to Merchant Store. Avail Bal: [BALANCE].",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "upi_transaction",
        "description": "Sanitized transactional entity tokens"
    },

    # -------------------------------------------------------------
    # BASE 4: Parcel Delivery Notice (HAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_DEL_01",
        "perturbation_id": "ROB_DEL_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Your package BD9284710 is out for delivery today. Delivery agent will arrive by 6 PM. Track at https://track.delivery.example.com.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "delivery_tracking",
        "description": "Standard parcel logistics notification baseline"
    },
    {
        "base_id": "BASE_DEL_01",
        "perturbation_id": "ROB_DEL_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "YOUR PACKAGE BD9284710 IS OUT FOR DELIVERY TODAY. DELIVERY AGENT WILL ARRIVE BY 6 PM. TRACK AT HTTPS://TRACK.DELIVERY.EXAMPLE.COM.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "delivery_tracking",
        "description": "All uppercase delivery notice"
    },
    {
        "base_id": "BASE_DEL_01",
        "perturbation_id": "ROB_DEL_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "Your package BD9284710 is out for delivery today !! Delivery agent will arrive by 6 PM ... Track at https://track.delivery.example.com .",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "delivery_tracking",
        "description": "Whitespace padding and punctuation noise"
    },
    {
        "base_id": "BASE_DEL_01",
        "perturbation_id": "ROB_DEL_01_PARA",
        "perturbation_type": "benign_paraphrase",
        "sms": "Shipment BD9284710 is scheduled for delivery to your address this afternoon before 6 PM. Live tracking available on our official portal.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "delivery_tracking",
        "description": "Natural rephrasing without explicit web link"
    },
    {
        "base_id": "BASE_DEL_01",
        "perturbation_id": "ROB_DEL_01_REDACT",
        "perturbation_type": "redaction_markers",
        "sms": "Your package [TRACKING_ID] is out for delivery today. Delivery agent will arrive by [TIME]. Track at [URL].",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "delivery_tracking",
        "description": "Sanitized delivery placeholders"
    },

    # -------------------------------------------------------------
    # BASE 5: Peer Informal Conversation (HAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_PER_01",
        "perturbation_id": "ROB_PER_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Hey, are you free this evening around 7pm to meet up for dinner and review the project slides?",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "personal_conversation",
        "description": "Casual peer social check-in baseline"
    },
    {
        "base_id": "BASE_PER_01",
        "perturbation_id": "ROB_PER_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "HEY, ARE YOU FREE THIS EVENING AROUND 7PM TO MEET UP FOR DINNER AND REVIEW THE PROJECT SLIDES?",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "personal_conversation",
        "description": "All uppercase peer message"
    },
    {
        "base_id": "BASE_PER_01",
        "perturbation_id": "ROB_PER_01_LOWER",
        "perturbation_type": "casing_all_lower",
        "sms": "hey, are you free this evening around 7pm to meet up for dinner and review the project slides?",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "personal_conversation",
        "description": "All lowercase peer message"
    },
    {
        "base_id": "BASE_PER_01",
        "perturbation_id": "ROB_PER_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "Hey , are you free this evening around 7pm to meet up for dinner and review the project slides ??? let me know !",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "personal_conversation",
        "description": "Punctuation spacing and multiple question marks"
    },
    {
        "base_id": "BASE_PER_01",
        "perturbation_id": "ROB_PER_01_HINGLISH",
        "perturbation_type": "code_switching_equivalence",
        "sms": "Bhai, aaj sham 7 baje free hai kya dinner ke liye? Project slides bhi check kar lenge.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "personal_conversation",
        "description": "Natural Hinglish equivalent of dinner coordination"
    },

    # -------------------------------------------------------------
    # BASE 6: Account/Appointment Notice (HAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_APT_01",
        "perturbation_id": "ROB_APT_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Appointment confirmation: Dr. Sharma at City Care Clinic on 18-Oct at 11:30 AM. Reply CANCEL if unable to attend.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "appointment_account",
        "description": "Clinical appointment reminder baseline"
    },
    {
        "base_id": "BASE_APT_01",
        "perturbation_id": "ROB_APT_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "APPOINTMENT CONFIRMATION: DR. SHARMA AT CITY CARE CLINIC ON 18-OCT AT 11:30 AM. REPLY CANCEL IF UNABLE TO ATTEND.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "appointment_account",
        "description": "All uppercase appointment notice"
    },
    {
        "base_id": "BASE_APT_01",
        "perturbation_id": "ROB_APT_01_PARA",
        "perturbation_type": "benign_paraphrase",
        "sms": "Your consultation with Dr. Sharma at City Care Clinic is confirmed for Oct 18, 11:30 AM. Text CANCEL if you need to reschedule.",
        "true_label": 0,
        "semantic_label_unchanged": True,
        "category": "appointment_account",
        "description": "Alternative medical scheduler rephrasing"
    },

    # -------------------------------------------------------------
    # BASE 7: Banking Credential Phishing (SPAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_PHISH_01",
        "perturbation_id": "ROB_PHISH_01_BASE",
        "perturbation_type": "baseline",
        "sms": "ALERT: Your bank account has been suspended due to unusual activity. Verify your identity immediately at https://security-verify-bank.com to avoid permanent closure.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "phishing_credential_theft",
        "description": "High urgency banking smishing baseline"
    },
    {
        "base_id": "BASE_PHISH_01",
        "perturbation_id": "ROB_PHISH_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "ALERT: YOUR BANK ACCOUNT HAS BEEN SUSPENDED DUE TO UNUSUAL ACTIVITY. VERIFY YOUR IDENTITY IMMEDIATELY AT HTTPS://SECURITY-VERIFY-BANK.COM TO AVOID PERMANENT CLOSURE.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "phishing_credential_theft",
        "description": "Aggressive all caps phishing alert"
    },
    {
        "base_id": "BASE_PHISH_01",
        "perturbation_id": "ROB_PHISH_01_LOWER",
        "perturbation_type": "casing_all_lower",
        "sms": "alert: your bank account has been suspended due to unusual activity. verify your identity immediately at https://security-verify-bank.com to avoid permanent closure.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "phishing_credential_theft",
        "description": "All lowercase phishing alert"
    },
    {
        "base_id": "BASE_PHISH_01",
        "perturbation_id": "ROB_PHISH_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "ALERT !! Your bank account has been suspended due to unusual activity ... Verify your identity IMMEDIATELY at https://security-verify-bank.com to avoid permanent closure !!!",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "phishing_credential_theft",
        "description": "Aggressive exclamation and ellipsis noise"
    },
    {
        "base_id": "BASE_PHISH_01",
        "perturbation_id": "ROB_PHISH_01_URG",
        "perturbation_type": "urgency_perturbation",
        "sms": "Notice: Your bank account has restricted access pending identity confirmation. Please verify your details at https://security-verify-bank.com within 48 hours to restore full access.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "phishing_credential_theft",
        "description": "Softened urgency phrasing to test evasion of threat triggers"
    },
    {
        "base_id": "BASE_PHISH_01",
        "perturbation_id": "ROB_PHISH_01_REDACT",
        "perturbation_type": "redaction_markers",
        "sms": "ALERT: Your bank account has been suspended due to unusual activity. Verify your identity immediately at [LINK] to avoid permanent closure.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "phishing_credential_theft",
        "description": "Defanged/redacted phishing link token"
    },

    # -------------------------------------------------------------
    # BASE 8: Fake Lottery / Prize Scam (SPAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_PRIZE_01",
        "perturbation_id": "ROB_PRIZE_01_BASE",
        "perturbation_type": "baseline",
        "sms": "CONGRATULATIONS! You have been selected as the winner of a $5,000 Walmart gift card! Call 09066364589 now or visit http://gift-claim-online.biz to claim your prize.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Classic advance-fee gift card lottery scam baseline"
    },
    {
        "base_id": "BASE_PRIZE_01",
        "perturbation_id": "ROB_PRIZE_01_LOWER",
        "perturbation_type": "casing_all_lower",
        "sms": "congratulations! you have been selected as the winner of a $5,000 walmart gift card! call 09066364589 now or visit http://gift-claim-online.biz to claim your prize.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "All lowercase prize spam"
    },
    {
        "base_id": "BASE_PRIZE_01",
        "perturbation_id": "ROB_PRIZE_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "CONGRATULATIONS !!! You have been selected as the WINNER of a $5,000 Walmart gift card !! Call 09066364589 now or visit http://gift-claim-online.biz to claim your prize ...",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Noisy punctuation and spacing"
    },
    {
        "base_id": "BASE_PRIZE_01",
        "perturbation_id": "ROB_PRIZE_01_PARA",
        "perturbation_type": "benign_paraphrase",
        "sms": "Great news! Your phone number won a $5,000 shopping voucher in our seasonal promotion. Phone 09066364589 immediately or visit our website http://gift-claim-online.biz to collect your voucher.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Synonymous prize claim phrasing"
    },
    {
        "base_id": "BASE_PRIZE_01",
        "perturbation_id": "ROB_PRIZE_01_REDACT",
        "perturbation_type": "redaction_markers",
        "sms": "CONGRATULATIONS! You have been selected as the winner of a [AMOUNT] Walmart gift card! Call [PHONE] now or visit [URL] to claim your prize.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Redacted prize contact entities"
    },

    # -------------------------------------------------------------
    # BASE 9: KYC Account Blocking Scam (SPAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_KYC_01",
        "perturbation_id": "ROB_KYC_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Dear Customer, your SBI account KYC has expired. Your debit card and netbanking will be blocked today. Click http://sbi-kyc-verification.in to update your PAN immediately.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "kyc_account_blocking",
        "description": "Bank KYC deactivation threat baseline"
    },
    {
        "base_id": "BASE_KYC_01",
        "perturbation_id": "ROB_KYC_01_CAPS",
        "perturbation_type": "casing_all_caps",
        "sms": "DEAR CUSTOMER, YOUR SBI ACCOUNT KYC HAS EXPIRED. YOUR DEBIT CARD AND NETBANKING WILL BE BLOCKED TODAY. CLICK HTTP://SBI-KYC-VERIFICATION.IN TO UPDATE YOUR PAN IMMEDIATELY.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "kyc_account_blocking",
        "description": "All uppercase KYC scam threat"
    },
    {
        "base_id": "BASE_KYC_01",
        "perturbation_id": "ROB_KYC_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "Dear Customer , your SBI account KYC has expired !! Your debit card and netbanking will be BLOCKED today ... Click http://sbi-kyc-verification.in to update your PAN immediately !!",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "kyc_account_blocking",
        "description": "Spaced punctuation and emphasis"
    },
    {
        "base_id": "BASE_KYC_01",
        "perturbation_id": "ROB_KYC_01_PARA",
        "perturbation_type": "benign_paraphrase",
        "sms": "Important notification: SBI account KYC verification is pending. Card and online banking services will freeze by end of day. Visit http://sbi-kyc-verification.in to complete document submission.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "kyc_account_blocking",
        "description": "Subtle administrative phrasing evasion test"
    },
    {
        "base_id": "BASE_KYC_01",
        "perturbation_id": "ROB_KYC_01_HINGLISH",
        "perturbation_type": "code_switching_equivalence",
        "sms": "Priy grahak, aapka SBI account KYC expire ho gaya hai. Aaj raat debit card block ho jayega. Turant update kare http://sbi-kyc-verification.in pe apna PAN card.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "kyc_account_blocking",
        "description": "Parallel Romanized Hindi (Hinglish) KYC threat"
    },

    # -------------------------------------------------------------
    # BASE 10: Fake Work From Home / Part-Time Job Scam (SPAM)
    # -------------------------------------------------------------
    {
        "base_id": "BASE_JOB_01",
        "perturbation_id": "ROB_JOB_01_BASE",
        "perturbation_type": "baseline",
        "sms": "Urgent Part-time job opening! Earn Rs. 2,000 - 5,000 daily working 1-2 hours from mobile. No experience needed. Contact HR manager on WhatsApp: +91-9876543210.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Part time daily wage recruitment scam baseline"
    },
    {
        "base_id": "BASE_JOB_01",
        "perturbation_id": "ROB_JOB_01_LOWER",
        "perturbation_type": "casing_all_lower",
        "sms": "urgent part-time job opening! earn rs. 2,000 - 5,000 daily working 1-2 hours from mobile. no experience needed. contact hr manager on whatsapp: +91-9876543210.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "All lowercase recruitment scam"
    },
    {
        "base_id": "BASE_JOB_01",
        "perturbation_id": "ROB_JOB_01_PUNCT",
        "perturbation_type": "whitespace_punctuation",
        "sms": "Urgent Part-time job opening !!! Earn Rs. 2,000 - 5,000 daily working 1-2 hours from mobile .. No experience needed !! Contact HR manager on WhatsApp: +91-9876543210",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Irregular punctuation and exclamation noise"
    },
    {
        "base_id": "BASE_JOB_01",
        "perturbation_id": "ROB_JOB_01_REDACT",
        "perturbation_type": "redaction_markers",
        "sms": "Urgent Part-time job opening! Earn [AMOUNT] daily working [HOURS] from mobile. No experience needed. Contact HR manager on WhatsApp: [PHONE].",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Redacted remuneration and phone markers"
    },
    {
        "base_id": "BASE_JOB_01",
        "perturbation_id": "ROB_JOB_01_HINGLISH",
        "perturbation_type": "code_switching_equivalence",
        "sms": "Ghar baithe kamaye daily 2000 se 5000 rupaye sirf mobile se 1-2 ghante me. Koi qualification nahi chahiye. Abhi WhatsApp kare: +91-9876543210.",
        "true_label": 1,
        "semantic_label_unchanged": True,
        "category": "fake_jobs_loans_prizes",
        "description": "Romanized Hindi daily income scam"
    },
]


def save_robustness_fixtures():
    out_dir = PROJECT_ROOT / "data" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures_csv = out_dir / "robustness_fixtures.csv"

    fieldnames = [
        "perturbation_id",
        "base_id",
        "perturbation_type",
        "sms",
        "true_label",
        "semantic_label_unchanged",
        "category",
        "description",
    ]

    with open(fixtures_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROBUSTNESS_FIXTURES)

    print(f"Saved {len(ROBUSTNESS_FIXTURES)} robustness fixtures to: {fixtures_csv}")
    return fixtures_csv


def run_robustness_evaluation():
    fixtures_csv = save_robustness_fixtures()

    models = ["full_finetuning_v2", "lora_v2", "dora_v2"]
    model_labels = {
        "full_finetuning_v2": "Full Fine-Tuning v2",
        "lora_v2": "LoRA v2 (r=8)",
        "dora_v2": "DoRA v2 (r=8, use_dora=True)",
    }

    # Group fixtures by base_id
    base_map = {}
    for f in ROBUSTNESS_FIXTURES:
        if f["perturbation_type"] == "baseline":
            base_map[f["base_id"]] = f

    results_payload = {
        "fixture_file": "data/robustness/robustness_fixtures.csv",
        "total_fixtures": len(ROBUSTNESS_FIXTURES),
        "total_base_messages": len(base_map),
        "models_robustness": {},
    }

    print("\n" + "=" * 80)
    print("RUNNING ROBUSTNESS EVALUATION ACROSS FROZEN MODELS")
    print("=" * 80)

    for m in models:
        threshold = get_default_threshold_for_method(m, str(PROJECT_ROOT))
        print(f"\nEvaluating: {model_labels[m]} | Threshold: {threshold:.2f}")

        # First evaluate all baseline messages
        base_preds = {}
        for base_id, base_item in base_map.items():
            res = predict_message(base_item["sms"], m, str(PROJECT_ROOT), threshold=threshold)
            pred = 1 if res["predicted_label"] == "SPAM" else 0
            base_preds[base_id] = {
                "pred": pred,
                "spam_prob": res["spam_probability"],
                "ham_prob": res["ham_probability"],
            }

        # Now evaluate all perturbations
        perturbation_results = []
        flips_count = 0
        score_deltas = []
        by_type_stats: Dict[str, Dict[str, Any]] = {}

        for item in ROBUSTNESS_FIXTURES:
            base_id = item["base_id"]
            ptype = item["perturbation_type"]
            base_info = base_preds[base_id]

            res = predict_message(item["sms"], m, str(PROJECT_ROOT), threshold=threshold)
            current_pred = 1 if res["predicted_label"] == "SPAM" else 0
            current_spam_prob = res["spam_probability"]

            is_flip = (current_pred != base_info["pred"])
            prob_delta = abs(current_spam_prob - base_info["spam_prob"])

            if ptype != "baseline":
                if is_flip:
                    flips_count += 1
                score_deltas.append(prob_delta)

                if ptype not in by_type_stats:
                    by_type_stats[ptype] = {
                        "count": 0,
                        "flips": 0,
                        "deltas": [],
                    }
                by_type_stats[ptype]["count"] += 1
                if is_flip:
                    by_type_stats[ptype]["flips"] += 1
                by_type_stats[ptype]["deltas"].append(prob_delta)

            perturbation_results.append({
                "perturbation_id": item["perturbation_id"],
                "base_id": base_id,
                "perturbation_type": ptype,
                "true_label": item["true_label"],
                "base_prediction": base_info["pred"],
                "perturbed_prediction": current_pred,
                "base_spam_prob": round(base_info["spam_prob"], 4),
                "perturbed_spam_prob": round(current_spam_prob, 4),
                "prob_delta": round(prob_delta, 4),
                "is_class_flip": is_flip,
                "semantic_unchanged": item["semantic_label_unchanged"],
            })

        total_non_baseline = len(score_deltas)
        consistency_rate = (total_non_baseline - flips_count) / total_non_baseline if total_non_baseline > 0 else 1.0
        mean_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
        max_delta = max(score_deltas) if score_deltas else 0.0

        print(f"  Overall Consistency Rate: {consistency_rate * 100:.2f}% ({total_non_baseline - flips_count}/{total_non_baseline})")
        print(f"  Class Flips Detected    : {flips_count}/{total_non_baseline}")
        print(f"  Mean Score Shift (|dP|) : {mean_delta:.4f}")
        print(f"  Max Score Shift (|dP|)  : {max_delta:.4f}")

        type_breakdown = {}
        for pt, st in sorted(by_type_stats.items()):
            pt_consist = (st["count"] - st["flips"]) / st["count"] if st["count"] > 0 else 1.0
            pt_mean_delta = sum(st["deltas"]) / len(st["deltas"]) if st["deltas"] else 0.0
            type_breakdown[pt] = {
                "count": st["count"],
                "flips": st["flips"],
                "consistency_rate": round(pt_consist, 4),
                "mean_score_shift": round(pt_mean_delta, 4),
                "max_score_shift": round(max(st["deltas"]) if st["deltas"] else 0.0, 4),
            }
            print(f"    - {pt:28s}: {st['count'] - st['flips']}/{st['count']} consistent ({pt_consist*100:5.1f}%) | mean dP: {pt_mean_delta:.4f}")

        results_payload["models_robustness"][m] = {
            "model_name": model_labels[m],
            "frozen_threshold": threshold,
            "overall_consistency_rate": round(consistency_rate, 4),
            "class_flips_count": flips_count,
            "total_perturbed_evals": total_non_baseline,
            "mean_score_shift": round(mean_delta, 4),
            "max_score_shift": round(max_delta, 4),
            "perturbation_type_breakdown": type_breakdown,
            "details": perturbation_results,
        }

    # Save to results/phase5/robustness_metrics.json
    out_json = PROJECT_ROOT / "results" / "phase5" / "robustness_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Robustness evaluation complete. Results saved to: {out_json}")
    print("=" * 80)
    return results_payload


if __name__ == "__main__":
    run_robustness_evaluation()
