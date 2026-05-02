# Sales Forecasting Website Fixes - Task Progress

## Approved Plan Summary
- **Sidebar**: Replace "Sales Prediction" → "Sales AI Platform" in base.html
- **Charts**: Add demo/sample data rendering in custom_dashboard.html when DB empty (total==0)
- **Components**: Goal Seek/Compare/Scenario - verify forms/endpoints (likely work post-data)
- **Data**: Seed demo predictions for charts/testing
- **Testing**: pip install → run app → login → predict → verify all

## Steps to Complete

### 1. Create/Edit TODO.md ✅ (Done)
### 2. Edit templates/base.html - Fix sidebar label
   - Replace both "Sales Prediction" instances
### 3. Edit templates/custom_dashboard.html - Charts with demo data
   - Modify `{% if total > 0 %}` to always render charts
   - Add fallback demo data in JS (trend, buckets)
### 4. Add demo data endpoint in app.py
   - `/seed_demo` route: Insert 20 sample predictions for current user
### 5. Edit database.py - Add seed_demo_data function
### 6. Test components:
   - Login → /goal_seek → submit form
   - /compare_models → submit
   - /scenario_page → submit
### 7. Verify Analytics charts: Visit /custom_dashboard → see rendered charts
### 8. Run full test: `python app.py` → browser → all features working
### 9. Update TODO.md with completion ✅
### 10. attempt_completion

**Next Step**: Edit base.html sidebar

