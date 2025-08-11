# Customer Revenue Analysis - SQL & Tableau

## Overview
This project analyzes customer behavior and revenue trends for a **DVD rental company** using the Sakila sample database.  
The goal was to design and implement a **Customer Revenue Data Warehouse** in MySQL and visualize insights in **Tableau** for data-driven decision-making.

## Project Files
- `sakila-schema.sql` – Creates the database schema.  
- `sakila-data.sql` – Inserts sample transactional data.  
- `customer_revenue_summary.sql` – Custom SQL view consolidating customer rental and revenue metrics.  
- `Final_Project_3H_cis467.docx` / `.pdf` – Project report with queries, explanations, and outputs.  
- `Team_3H.twbx` – Tableau workbook containing visualizations and dashboard.  
- `/images/` – PNG exports of Tableau visualizations and dashboard.

## Tools & Technologies
- **SQL** – MySQL (data warehouse creation, queries)  
- **Tableau** – Data visualization and dashboard creation  
- **Sakila Database** – Sample DVD rental dataset  

## Analysis & Insights
The SQL view `customer_revenue_summary` aggregates:
- Total rentals & revenue per customer  
- Average spending per rental  
- Most frequent movie category  
- Total late fees  
- First & last rental dates  
- Customer location  

### Insights
1. **VIP Customers** - High-spending, frequent renters identified for loyalty programs.  
2. **Regional Spending Trends** – Countries ranked by average spend per rental.  
3. **Monthly Revenue Trends** – Seasonal demand fluctuations visualized.  
4. **Customer Churn** – Inactive customers flagged for re-engagement campaigns.  
5. **Top Genres by Revenue** – Most profitable genres per country and overall.

## 📷 Visualizations
![Global Revenue Map](<img width="1326" height="1029" alt="Graph1" src="https://github.com/user-attachments/assets/cd9c7eb2-b1b8-459a-b265-9891a71108bb" />)
![Genre Revenue Bubble Chart](images/genre_revenue.png)  
![Dashboard](images/dashboard.png)  

## License
This project is licensed under the [MIT License](LICENSE).
