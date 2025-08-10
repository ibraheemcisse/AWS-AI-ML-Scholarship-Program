# AI-Powered Property Management – Payment Analysis (PartyRock App)

This project is a **PartyRock** app module designed to analyze tenant payment behavior, detect patterns, and estimate a risk score for landlords and property managers.

It is part of a larger AI-powered property management system, which can include modules for:
- Payment Analysis
- Tenant Risk Evaluation
- Maintenance Prioritization
- Automated Listing Generation

---

## How It Works
1. You enter tenant payment history and notes.
2. The AI analyzes payment reliability, behavioral patterns, and assigns a risk score (0–10).
3. The app recommends landlord actions based on the analysis.

---

## How to Use
1. Open [AWS PartyRock](https://partyrock.aws/).
2. Click **Import App**.
3. Copy and paste the JSON below into the importer.
4. Test with your own tenant data or the example provided.

---

## Example Input
Tenant: John Doe
Monthly Rent: $1,200
Payments:

Jan 2025: Paid on time

Feb 2025: Paid 3 days late

Mar 2025: Paid on time
Notes: Generally communicative, occasional short delays.

swift
Copy
Edit

---

## PartyRock JSON
```json
{
  "widgets": [
    {
      "id": "tenantData",
      "type": "multiLineTextInput",
      "name": "Tenant Payment Data",
      "placeholder": "Enter tenant name, rent amount, payment dates, and behavior notes"
    },
    {
      "id": "paymentAnalysis",
      "type": "textGeneration",
      "name": "Payment Behavior Analysis",
      "inputMapping": {
        "input": "{{tenantData}}"
      },
      "prompt": "Analyze the following tenant payment behavior:\n\n{{tenantData}}\n\nProvide:\n- Payment reliability assessment\n- Observed patterns\n- Potential risk score (0-10)\n- Recommended landlord action"
    }
  ],
  "layout": {
    "rows": [
      ["tenantData"],
      ["paymentAnalysis"]
    ]
  },
  "metadata": {
    "name": "Payment Analysis Module",
    "description": "Analyzes tenant payment behavior and estimates risk score.",
    "author": "Your Name",
    "version": "0.1.0"
  }
}
License
MIT

yaml
Copy
Edit

---

If you want, I can also make a **matching one-page markdown** for each module so you end up with four self-contained PartyRock app files. That way the whole property management system is modular.








Ask ChatGPT
