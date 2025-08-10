# Nomad AI Concierge – Travel Planner & Visa Advisor (PartyRock App)

Nomad AI Concierge is an **AI-powered travel assistant** tailored for backpackers and digital nomads. It helps craft personalized itineraries, shed light on visa requirements, highlight hidden local events, and assist with budget tracking—all via a JSON-driven UI.:contentReference[oaicite:0]{index=0}

---

## How It Works
- Select your travel mode (Backpacker or Digital Nomad), destination, travel dates, interests, and passport country.
- The AI generates:
  - A bespoke **travel itinerary**
  - **Visa requirements** & documentation checklist
  - Suggested **local events & hidden gems**
  - A **budget plan** with daily spend tracking 

---

## How to Use It
1. Visit [AWS PartyRock](https://partyrock.aws/).
2. Click **Import App**.
3. Copy and paste the JSON below.
4. Input your travel preferences and enjoy instant planning.

---

## PartyRock JSON Configuration
```json
{
  "title": "Nomad AI Concierge – Travel Planner & Visa Advisor",
  "description": "An AI-powered personalized travel assistant for Backpackers and Digital Nomads to generate travel itineraries, check visa requirements, suggest local events, and track budgets.",
  "widgets": [
    {
      "type": "select",
      "id": "user_type",
      "name": "Select Travel Mode",
      "options": [
        { "label": "Backpacker", "value": "backpacker" },
        { "label": "Digital Nomad", "value": "digital_nomad" }
      ],
      "default": "backpacker"
    },
    {
      "type": "text-input",
      "id": "destination_input",
      "name": "Destination",
      "placeholder": "e.g. Bali, Indonesia"
    },
    {
      "type": "text-input",
      "id": "dates_input",
      "name": "Travel Dates",
      "placeholder": "YYYY-MM-DD to YYYY-MM-DD"
    },
    {
      "type": "text-input",
      "id": "interests_input",
      "name": "Interests (comma separated)",
      "placeholder": "e.g. hiking, coworking, cultural events"
    },
    {
      "type": "text-input",
      "id": "passport_country_input",
      "name": "Passport Country",
      "placeholder": "e.g. USA"
    },
    {
      "type": "text-generator",
      "id": "itinerary_output",
      "name": "Personalized Travel Itinerary",
      "promptTemplate": "Create a detailed {{user_type}} travel itinerary for visiting {{destination_input}} from {{dates_input}} focusing on these interests: {{interests_input}}."
    },
    {
      "type": "text-generator",
      "id": "visa_requirements_output",
      "name": "Visa Requirements & Checklist",
      "promptTemplate": "Provide visa requirements and a documentation checklist for a traveler holding a passport from {{passport_country_input}} visiting {{destination_input}}."
    },
    {
      "type": "text-generator",
      "id": "local_events_output",
      "name": "Local Events & Hidden Gems",
      "promptTemplate": "Suggest upcoming local events and hidden gems in {{destination_input}} relevant to someone interested in {{interests_input}}."
    },
    {
      "type": "number-input",
      "id": "budget_input",
      "name": "Total Travel Budget (USD)",
      "placeholder": "e.g. 1500"
    },
    {
      "type": "text-generator",
      "id": "budget_plan_output",
      "name": "Budget Tracker & Daily Spend Plan",
      "promptTemplate": "Create a budget plan and daily spending tracker for a total budget of ${{budget_input}} over the trip from {{dates_input}}."
    }
  ]
}



