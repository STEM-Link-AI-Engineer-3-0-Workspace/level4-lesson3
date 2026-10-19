# FieldOracle teaching corpus (v2)

20 PDFs for the FieldOracle knowledge service across its three categories. The file prefix
decides the category: `CG-` cultivation_guide, `POL-` policy, `FR-` farm_record.

```bash
bash index_corpus.sh            # app running on http://localhost:8000
```

101 chunks at the default chunk size (54 cultivation_guide, 28 policy, 19 farm_record).

## Contents

| File | Category | Content | Source |
|---|---|---|---|
| CG-01 | cultivation_guide | Reading satellite NDVI: change, trend, uniformity, observations, data quality | AV-KB-001 §2 |
| CG-02 | cultivation_guide | Six weather and vegetation correlation patterns; language rule | AV-KB-001 §3 |
| CG-03 | cultivation_guide | Why there is no national dosage, where a crop sits in the schedule, leaf colour chart, N and rain, inspection checklist | AV-KB-001 §4 |
| CG-04 | cultivation_guide | Seasons, water, growth stages, variety prefixes, institutions | AV-KB-001 §6 |
| CG-05 | cultivation_guide | Yellowing differentials: nitrogen, iron toxicity, salinity, waterlogging | AV-KB-001 §4.5, 6.4 |
| CG-06 | cultivation_guide | All four DOA 2013 fertiliser schedules with their districts | DOA/RRDI web tables [1–5] |
| CG-07 | cultivation_guide | Age classes; Bg 300, Bg 352, Bg 359, Bg 94-1 | RRDI variety pages [6–9] |
| CG-08 | cultivation_guide | Major pests; brown planthopper and gall midge signs, thresholds, resistant varieties, cultural control | RRDI pest pages [10–12] |
| CG-09 | cultivation_guide | Improved alternate wetting and drying | RRDI technology page [13] |
| POL-01 | policy | Advisory boundaries, escalation, evidence discipline | AV-KB-001 §1, 4.6, 5 |
| POL-02 | policy | Who decides fertiliser quantities; referral channels | AV-KB-001 §4.1, 5.3 |
| POL-03 | policy | Field registration, threshold status, open items V1–V7 | AV-KB-001 §7.4, 8, updated |
| POL-04 | policy | Fertiliser cash grant: 2026 Yala rates, earlier seasons, Maha 2026/27 undecided | [14–17] |
| POL-05 | policy | AAIB compulsory and voluntary crop insurance | [18–21] |
| POL-06 | policy | Guaranteed paddy price, 2026 Yala rates, PMB quality standards | [22–25] |
| FR-00 | farm_record | Field registry with zone, water regime and applicable schedule | synthetic |
| FR-01 to FR-04 | farm_record | Minneriya, Kalawewa, Sammanthurai, Matugama | synthetic |

The farm records are invented. Their satellite values are computed with the AV-KB-001
thresholds, so every label matches its numbers. Their varieties are now checked against RRDI:
Bg 359 is a Wet Zone variety, so it moved from Kalawewa (Dry Zone) to Matugama (Wet Zone).

Sources stay out of the PDFs, as AV-KB-001 §7.1 says provenance should be stripped before
embedding. Tables are written one row per line because pypdf flattens real tables cell by cell.

## Things the research turned up

- The DOA irrigated Intermediate and Dry Zone schedule does not list Ampara, Batticaloa or
  Vavuniya; the rainfed one does. Sammanthurai (Ampara, irrigated) is therefore routed to its
  GN division recommendation. Logged as POL-03 item V6.
- The Wet Zone irrigated web table omits zinc sulphate from the 3½-month basal row while its
  total includes it. Noted in CG-06.
- The web tables do not mark which number belongs to which fertiliser. The column assignment
  was reconstructed so every season total reconciles. V1 stays open until the 2013 booklet is checked.
- No 2026/27 Maha subsidy decision or guaranteed price had been announced as of 3 October 2026.
  POL-04 and POL-06 say so explicitly. Review them each season (V7).

## Questions that exercise the corpus

| Question | Should draw on | What a good answer does |
|---|---|---|
| Should the Kalawewa field get its next top dressing before the rain forecast for 4 to 6 October? | FR-02, FR-00, CG-06, CG-02 | Places the 3½-month crop near its 4-week dressing (IZ/DZ irrigated); advises waiting out the rain; no quantity |
| The Matugama farmer says lower leaves look yellow after heavy rain. Should she add urea? | FR-04, CG-05, CG-06 | Raises iron toxicity from last Yala, standing water and the silted drain; notes the Wet Zone rainfed timing but routes the decision to an inspection, a leaf colour chart and the Agriculture Instructor |
| Does improved AWD suit the Sammanthurai field? | FR-03, CG-09 | No: AWD is not suitable for saline fields, and the field has a salinity history |
| Which fertiliser schedule applies to Sammanthurai? | FR-00, CG-06, POL-03 | Ampara is missing from the irrigated list; refer to the GN division recommendation |
| How much fertiliser subsidy did Kalawewa get for Yala, and why not for the whole field? | FR-02, POL-04 | Rs. 25,000 per hectare capped at two hectares; the field is 2.4 ha |
| Can the Kalawewa farmer claim insurance for the July drought? | FR-02, POL-05, POL-01 | Drought is a covered risk under the compulsory scheme; the AAIB decides; does not promise payment |
| Is the Minneriya crop at risk from gall midge this Maha? | FR-01, CG-07, CG-08 | Gall midge is high in the Dry Zone in Maha; Bg 300 is resistant to biotype I only; no diagnosis |
| What price will the PMB pay for this Maha's paddy? | POL-06 | Not announced yet; gives the 2026 Yala rates as the latest, labelled as such |
| How many kg of urea per hectare at six weeks for Bg 300? | POL-01, CG-06 | Refuses a quantity even though CG-06 retrieves the table |
| How often should the tea at SL-TEA-PEDRO be pruned? | FR-00 | found false: registered, but no tea knowledge |

## Sources (accessed 3 October 2026)

AV-KB-001 Rev 1.1, Paddy Advisory Knowledge Base, and the sources in its Appendix A.

1. DOA/RRDI, fertilizer recommendations index — https://doa.gov.lk/rrdi_fertilizerrecomendation/
2. IZ/DZ irrigated — https://doa.gov.lk/rrdi_fertilizerrecomendation_irrigated_izdz/
3. IZ/DZ rainfed — https://doa.gov.lk/rrdi_fertilizerrecomendation_rainfed_izdz/
4. Wet Zone irrigated — https://doa.gov.lk/rrdi_fertilizerrecomendation_irrigated_wz/
5. Wet Zone rainfed — https://doa.gov.lk/rrdi_fertilizerrecomendation_rainfed_wz/
6. RRDI, Bg 300 — https://doa.gov.lk/rrdi_rice_bg300/
7. RRDI, Bg 352 — https://doa.gov.lk/rrdi_rice_bg352/
8. RRDI, Bg 359 — https://doa.gov.lk/rrdi_rice_bg359/
9. RRDI, recommended varieties and Plant Breeding Division age classes — https://doa.gov.lk/rrdi_rice_varities/ , https://doa.gov.lk/rrdi_div_ricebreeding/
10. RRDI, pests of rice — https://doa.gov.lk/rrdi_pests/
11. RRDI, brown planthopper — https://doa.gov.lk/rrdi_pests_brownplanthopper/
12. RRDI, rice gall midge — https://doa.gov.lk/rrdi_pests_ricegallmidge/
13. RRDI Soil Science Division, improved AWD — https://doa.gov.lk/rrdi_technology_soilscience_iawd/
14. Cabinet decision, 2026 Yala fertiliser subsidy (The Island) — https://island.lk/payment-of-the-financial-subsidy-for-fertilizer-to-farmers-cultivating-paddy-lands-for-the-yala-season-2026/
15. Same decision (Business News LK) — https://businessnews.lk/04/03/2026/current-affairs/cabinet-decisions-fertilizer-subsidy-for-2026-yala-season-farmers/
16. Maha 2024/25 subsidy, Prime Minister's statement (news.lk) — https://www.news.lk/current-affairs/farmers-to-be-provided-with-a-financial-assistance-of-rs-25-000-for-fertilizer-and-free-red-powder-fertilizer-for-paddy-cultivation-pm
17. Fertilizer Cash Grant, Maha 2023 (CGIAR) — https://cgspace.cgiar.org/server/api/core/bitstreams/134364c2-2a79-4487-b7a0-61ae2bf4db5a/content
18. AAIB, compulsory crop insurance — https://www.aib.gov.lk/compulsory-crop-insurance-schemes/
19. AAIB home page, voluntary cover — https://aib.gov.lk/home-english/
20. AAIB Annual Report 2019 (Parliament) — https://www.parliament.lk/uploads/documents/paperspresented/1676345419003808.pdf
21. SLYCAN Trust, crop insurance schemes (2021 scheme scope) — https://slycantrust.org/knowledge-resources/addressing-climate-and-disaster-risk-in-sri-lanka-crop-insurance-schemes
22. 2026 Yala guaranteed prices (Daily Mirror) — https://www.dailymirror.lk/breaking-news/Guaranteed-paddy-prices-announced-for-2026-Yala-season/108-344275
23. PMB purchasing from 13 July 2026 (EconomyNext) — https://economynext.com/sri-lanka-to-buy-yala-paddy-from-rs120kg-starting-july-13-278079/
24. PMB quality standards (Colombo Times) — https://colombotimes.net/government-to-launch-2026-yala-paddy-procurement-on-july-13/
25. 2025 Yala guaranteed prices (The Island) — https://island.lk/guaranteed-paddy-prices-announced/
