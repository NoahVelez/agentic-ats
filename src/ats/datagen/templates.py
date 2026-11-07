"""Career-field profiles: titles and industry for fact sheets, plus offline-renderer templates.

Template text must never contain PII, URLs, or emails. Bullets that contain a forbidden keyword
for a given candidate are filtered out at render time, so they need not be keyword-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldProfile:
    industry: str
    titles: tuple[str, ...]
    companies: tuple[str, ...]
    bullets: tuple[str, ...]
    skills: tuple[str, ...]
    projects: tuple[tuple[str, tuple[str, ...]], ...]
    degrees: tuple[tuple[str, str], ...]
    summaries: tuple[str, ...]
    # Words that tie a transferable bullet to this field (used to keep career changers coherent).
    markers: tuple[str, ...] = field(default_factory=tuple)


INSTITUTIONS: tuple[str, ...] = (
    "Lakeshore State University",
    "Northbridge University",
    "Cedar Valley College",
    "Riverside Technical Institute",
    "Granite Hill University",
    "Westmoor Community College",
    "Bayview State University",
    "Pinecrest College",
)

FIELDS: dict[str, FieldProfile] = {
    "analytics": FieldProfile(
        industry="analytics and business intelligence",
        titles=(
            "Senior Data Analyst",
            "Analytics Lead",
            "Product Analyst",
            "Business Intelligence Analyst",
            "Senior Analyst, Growth",
            "Insights Analyst",
            "Data Analyst",
        ),
        companies=(
            "Brightpath Retail",
            "Quillwave Software",
            "Harborline Insurance",
            "Copperleaf Health Plans",
            "Northwind Subscriptions",
            "Tidewater Logistics Tech",
            "Summitview Media",
            "Bluefern Payments",
        ),
        bullets=(
            "Forecasted quarterly subscription revenue within 3% of actuals for four quarters",
            "Cleaned and documented the customer master table used by billing and support",
            "Built cohort retention curves that changed how the product group ran onboarding",
            "Wrote metric definitions for activation and retention in the company data glossary",
            "Audited event tracking and fixed more than 20 broken signup events",
            "Partnered with finance on the annual planning model for headcount and spend",
            "Sized the market opportunity for two new sales regions",
            "Investigated a 12% drop in renewals and traced it to a billing bug",
            "Built a customer segmentation based on purchase frequency and order value",
            "Automated the monthly board metrics pack, saving two days of manual work",
            "Reviewed pull requests for the shared metrics repository",
            "Estimated lifetime value by acquisition channel to guide marketing spend",
        ),
        skills=(
            "Excel",
            "Looker",
            "statistics",
            "forecasting",
            "data modeling",
            "dbt",
            "Git",
            "segmentation",
            "cohort analysis",
        ),
        projects=(
            (
                "Churn early-warning model",
                ("Flagged at-risk accounts two months before renewal using usage signals",),
            ),
            (
                "Metrics glossary",
                ("Documented 60 company metrics with owners, formulas, and caveats",),
            ),
            (
                "Pricing elasticity study",
                ("Estimated price sensitivity for three plan tiers from historical upgrades",),
            ),
        ),
        degrees=(
            ("B.S.", "Statistics"),
            ("B.A.", "Economics"),
            ("M.S.", "Business Analytics"),
            ("B.S.", "Mathematics"),
        ),
        summaries=(
            "Analyst with {years} years of experience turning messy business data into decisions.",
            "Numbers-focused professional with {years} years in {industry}.",
            "{years} years of experience helping product and finance teams plan with data.",
        ),
    ),
    "marketing": FieldProfile(
        industry="marketing",
        titles=("Marketing Coordinator", "Social Media Specialist", "Content Associate"),
        companies=("Lumen & Co", "Redbird Creative", "Fieldstone Brands", "Orchard Media Group"),
        bullets=(
            "Scheduled social posts across four channels and kept the content calendar current",
            "Coordinated trade show logistics for booths in three cities",
            "Wrote product descriptions for 150 catalog items",
            "Managed influencer gifting shipments and tracked deliveries",
            "Drafted weekly newsletter copy for a list of 40,000 subscribers",
            "Organized the brand asset library so designers could find approved logos",
            "Booked venues and vendors for two customer appreciation events",
            "Proofread campaign emails before they went out",
            "Collected customer testimonials for the website refresh",
            "Ordered printed collateral and tracked the marketing supplies budget",
        ),
        skills=("Canva", "Hootsuite", "copywriting", "event planning", "Mailchimp", "proofreading"),
        projects=(
            ("Holiday campaign", ("Wrote the copy for a five-email holiday sequence",)),
            ("Brand refresh", ("Collected feedback on three logo options from customers",)),
        ),
        degrees=(("B.A.", "Communications"), ("B.A.", "Marketing")),
        summaries=(
            "Marketing professional with {years} years of experience in content and events.",
            "Creative coordinator with {years} years in {industry}.",
        ),
    ),
    "administration": FieldProfile(
        industry="office administration",
        titles=("Office Administrator", "Executive Assistant", "Administrative Coordinator"),
        companies=(
            "Stonegate Partners",
            "Maple Street Dental",
            "Crescent Legal Group",
            "Arbor Realty",
        ),
        bullets=(
            "Managed calendars and travel for three senior partners",
            "Processed vendor invoices and matched them to purchase orders",
            "Onboarded new hires with equipment, badges, and paperwork",
            "Maintained the office supply inventory and reorder schedule",
            "Answered a multi-line phone system and routed client calls",
            "Organized quarterly all-hands meetings for 80 employees",
            "Prepared meeting minutes and distributed action items",
            "Renegotiated the office cleaning contract and saved 12% annually",
            "Kept the shared file server organized with a clear folder structure",
            "Coordinated building maintenance requests with the landlord",
        ),
        skills=(
            "Microsoft Office",
            "scheduling",
            "Outlook",
            "bookkeeping",
            "data entry",
            "QuickBooks",
        ),
        projects=(
            ("Office move", ("Planned the move of 40 desks to a new floor over one weekend",)),
            ("Filing overhaul", ("Digitized ten years of paper client files",)),
        ),
        degrees=(("A.A.", "Business Administration"), ("B.A.", "English")),
        summaries=(
            "Organized administrative professional with {years} years of experience.",
            "Detail-oriented coordinator with {years} years in {industry}.",
        ),
    ),
    "healthcare": FieldProfile(
        industry="healthcare",
        titles=("Registered Nurse", "Charge Nurse", "Clinical Nurse"),
        companies=(
            "Mercy Valley Hospital",
            "St. Brigid Medical Center",
            "Harbor Point Clinic",
            "Elmwood Care",
        ),
        bullets=(
            "Provided bedside care for up to six patients per shift on a medical-surgical unit",
            "Administered medications and documented care in the electronic health record",
            "Educated patients on discharge instructions and follow-up care",
            "Responded to rapid-response calls and stabilized deteriorating patients",
            "Precepted new graduate nurses during their first 12 weeks",
            "Served on the unit infection-prevention committee",
            "Coordinated with pharmacy to resolve medication discrepancies",
            "Completed hourly safety rounds and fall-risk assessments",
        ),
        skills=("patient care", "BLS", "ACLS", "EHR documentation", "triage", "wound care"),
        projects=(
            ("Fall prevention initiative", ("Helped cut unit falls by 30% over one year",)),
            ("Discharge checklist", ("Co-wrote a discharge checklist adopted by two units",)),
        ),
        degrees=(("B.S.N.", "Nursing"), ("A.D.N.", "Nursing")),
        summaries=(
            "Registered nurse with {years} years of experience, now moving into analytics.",
            "Clinician with {years} years in {industry} seeking a data-focused role.",
        ),
        markers=(
            "nurse",
            "nurses",
            "nursing",
            "patient",
            "patients",
            "clinic",
            "hospital",
            "physicians",
            "triage",
            "bed",
        ),
    ),
    "education": FieldProfile(
        industry="K-12 education",
        titles=("High School Math Teacher", "Middle School Teacher", "Instructional Coach"),
        companies=(
            "Oak Ridge High School",
            "Westfield Middle School",
            "Lincoln Park Academy",
            "Riverbend Unified",
        ),
        bullets=(
            "Taught algebra and geometry to five sections of 30 students",
            "Wrote unit plans aligned to state standards",
            "Raised the pass rate on the state algebra exam from 61% to 78%",
            "Ran an after-school tutoring program for struggling students",
            "Advised the math club and coached the regional competition team",
            "Held parent conferences and wrote individualized learning plans",
            "Introduced weekly exit tickets to check understanding",
            "Organized the school science fair for 200 participants",
        ),
        skills=(
            "curriculum design",
            "classroom management",
            "Google Classroom",
            "tutoring",
            "Excel",
        ),
        projects=(
            ("Reading intervention tracker", ("Tracked weekly reading levels for 120 students",)),
            ("Math night", ("Organized a family math night attended by 150 parents",)),
        ),
        degrees=(("B.A.", "Mathematics Education"), ("M.Ed.", "Curriculum and Instruction")),
        summaries=(
            "Educator with {years} years of experience, transitioning into analytics.",
            "Teacher with {years} years in {industry} who loves working with data.",
        ),
        markers=(
            "student",
            "students",
            "teachers",
            "teaching",
            "classrooms",
            "principal",
            "grade",
            "reading",
        ),
    ),
    "military": FieldProfile(
        industry="military service",
        titles=("Platoon Leader", "Operations Officer", "Logistics Officer"),
        companies=("U.S. Army", "U.S. Army Reserve", "Army National Guard"),
        bullets=(
            "Planned and executed field training exercises for 120 personnel",
            "Maintained accountability for equipment valued at $4M",
            "Wrote operations orders for convoy movements across three states",
            "Trained soldiers on safety procedures and equipment maintenance",
            "Coordinated with adjacent units on shared range schedules",
            "Earned a commendation for readiness during a deployment rotation",
        ),
        skills=("operations planning", "risk assessment", "training", "Excel", "logistics"),
        projects=(("Readiness tracker", ("Built a spreadsheet tracker for equipment readiness",)),),
        degrees=(("B.S.", "History"), ("B.S.", "Systems Engineering")),
        summaries=(
            "Army veteran with {years} years of experience leading operations, now in analytics.",
            "Former officer with {years} years in {industry}.",
        ),
        markers=("platoon", "battalion", "soldiers", "commander", "command", "readiness"),
    ),
    "logistics": FieldProfile(
        industry="warehouse and distribution logistics",
        titles=("Warehouse Supervisor", "Distribution Coordinator", "Inventory Specialist"),
        companies=(
            "Keystone Distribution",
            "Rapidline Freight",
            "Coastal Fulfillment",
            "Midland Supply Co",
        ),
        bullets=(
            "Scheduled inbound trucks for a 200,000 square-foot distribution center",
            "Ran cycle counts and kept inventory accuracy above 99%",
            "Trained new pickers on safety and scanner procedures",
            "Cut late outbound shipments by 25% by rebalancing shift coverage",
            "Coordinated returns processing with three carriers",
            "Maintained forklift certification records for the floor staff",
        ),
        skills=("inventory control", "WMS", "Excel", "forklift certified", "OSHA safety"),
        projects=(("Slotting redesign", ("Moved fast-moving items closer to packing stations",)),),
        degrees=(("A.S.", "Supply Chain Management"), ("B.S.", "Business")),
        summaries=(
            "Operations professional with {years} years in {industry}, moving into analytics.",
            "Logistics coordinator with {years} years of hands-on experience.",
        ),
        markers=(
            "warehouse",
            "shipment",
            "shipments",
            "SKUs",
            "picks",
            "carrier",
            "distribution",
            "inventory",
        ),
    ),
    "culinary": FieldProfile(
        industry="restaurants and hospitality",
        titles=("Line Cook", "Sous Chef", "Pastry Cook"),
        companies=("The Copper Pot", "Harbor Street Bistro", "Juniper Kitchen", "Old Mill Tavern"),
        bullets=(
            "Prepared 300 covers per night on the saute station",
            "Developed four seasonal specials using local produce",
            "Trained new cooks on knife skills and plating standards",
            "Kept the walk-in organized and rotated stock daily",
            "Passed every health inspection during my time on the line",
            "Prepped sauces and stocks for the dinner service",
        ),
        skills=("knife skills", "menu development", "food safety", "plating", "baking"),
        projects=(("Seasonal menu", ("Created a five-course autumn tasting menu",)),),
        degrees=(("A.O.S.", "Culinary Arts"), ("Certificate", "Baking and Pastry")),
        summaries=("Cook with {years} years of experience in busy kitchens.",),
    ),
    "construction": FieldProfile(
        industry="residential construction",
        titles=("Carpenter", "Site Foreman", "Framing Lead"),
        companies=(
            "Ridgeview Builders",
            "Ironwood Construction",
            "Cornerstone Homes",
            "Pioneer Framing",
        ),
        bullets=(
            "Framed walls and roofs for 40 single-family homes",
            "Read blueprints and laid out foundations to spec",
            "Installed custom cabinetry and finish trim",
            "Kept the job site clean and compliant with safety rules",
            "Ordered lumber and fasteners for weekly builds",
            "Trained apprentices on power tool safety",
        ),
        skills=("framing", "blueprint reading", "finish carpentry", "OSHA 30", "estimating"),
        projects=(("Deck build", ("Designed and built a two-level cedar deck",)),),
        degrees=(("Certificate", "Carpentry"), ("A.A.S.", "Construction Technology")),
        summaries=("Skilled tradesperson with {years} years in {industry}.",),
    ),
    "design": FieldProfile(
        industry="graphic design",
        titles=("Graphic Designer", "Junior Designer", "Production Artist"),
        companies=("Inkwell Studio", "Paperkite Design", "Northlight Agency", "Foundry Print"),
        bullets=(
            "Designed packaging for a line of 12 specialty teas",
            "Created print layouts for a quarterly magazine",
            "Built brand style guides for three local businesses",
            "Retouched product photos for an online catalog",
            "Prepared print-ready files and checked proofs with vendors",
            "Illustrated icons for a mobile banking app",
        ),
        skills=("Adobe Illustrator", "Photoshop", "InDesign", "typography", "Figma"),
        projects=(("Poster series", ("Designed a six-poster series for a film festival",)),),
        degrees=(("B.F.A.", "Graphic Design"), ("A.A.", "Visual Communication")),
        summaries=("Designer with {years} years of experience in print and digital work.",),
    ),
    "retail": FieldProfile(
        industry="retail",
        titles=("Sales Associate", "Shift Supervisor", "Visual Merchandiser"),
        companies=("Greenway Outfitters", "Sunrise Home Goods", "Maple & Main", "Corner Books"),
        bullets=(
            "Helped customers find products and processed returns",
            "Set up seasonal window displays",
            "Opened and closed the store and balanced the register",
            "Restocked shelves after weekly deliveries",
            "Trained new associates on the point-of-sale system",
            "Earned top seller recognition three months in a row",
        ),
        skills=("customer service", "merchandising", "cash handling", "POS systems"),
        projects=(("Holiday display", ("Designed the main holiday window display",)),),
        degrees=(("High School Diploma", "General Studies"), ("A.A.", "Liberal Arts")),
        summaries=("Customer-focused associate with {years} years in {industry}.",),
    ),
}

QUALIFIED_ARCHETYPES = ("clear_fit", "hidden_gem", "near_miss")
STUFFER_FIELDS: tuple[str, ...] = ("marketing", "administration")
CHANGER_FIELDS: tuple[str, ...] = ("healthcare", "education", "military", "logistics")
FILLER_FIELDS: tuple[str, ...] = ("culinary", "construction", "design", "retail")
