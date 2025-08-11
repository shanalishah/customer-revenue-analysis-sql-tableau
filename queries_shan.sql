-- =====================================
-- CREATING DATA WAREHOUSE
-- =====================================
USE sakila;
DROP VIEW IF EXISTS customer_revenue_summary;

CREATE VIEW customer_revenue_summary AS
SELECT 
    c.customer_id AS Customer_ID,
    CONCAT(c.first_name, ' ', c.last_name) AS Customer_Name,
    COUNT(r.rental_id) AS Total_Rentals, 
    ROUND(SUM(p.amount), 2) AS Total_Revenue, 
    ROUND(SUM(p.amount) / COUNT(r.rental_id), 2) AS Avg_Spending_Per_Rental, 
    COUNT(DISTINCT fc.category_id) AS Total_Categories_Rented, 
    (SELECT name 
     FROM category 
     JOIN film_category fc ON category.category_id = fc.category_id
     JOIN inventory i ON fc.film_id = i.film_id
     JOIN rental r2 ON i.inventory_id = r2.inventory_id
     WHERE r2.customer_id = c.customer_id
     GROUP BY category.name
     ORDER BY COUNT(*) DESC
     LIMIT 1) AS Most_Frequent_Category, 
    ROUND(COALESCE(SUM(CASE 
        WHEN DATEDIFF(r.return_date, r.rental_date) > f.rental_duration 
        THEN (DATEDIFF(r.return_date, r.rental_date) - f.rental_duration) * 0.50 
        ELSE 0 END), 0), 2) AS Total_Late_Fees, 
    (SELECT MIN(rental_date) FROM rental WHERE customer_id = c.customer_id) AS First_Rental_Date, 
    (SELECT MAX(rental_date) FROM rental WHERE customer_id = c.customer_id) AS Last_Rental_Date, 
    COUNT(p.payment_id) AS Total_Payments,
    ci.city AS Customer_City, 
    co.country AS Customer_Country 
FROM customer c
JOIN address a ON c.address_id = a.address_id
JOIN city ci ON a.city_id = ci.city_id
JOIN country co ON ci.country_id = co.country_id
JOIN rental r ON c.customer_id = r.customer_id
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN film f ON i.film_id = f.film_id
LEFT JOIN payment p ON r.rental_id = p.rental_id
LEFT JOIN film_category fc ON i.film_id = fc.film_id
GROUP BY c.customer_id, Customer_Name, ci.city, co.country;

SELECT * FROM customer_revenue_summary;


-- QUERIES

-- 1 Top 10 Revenue-Generating Customers
SELECT Customer_Name, Total_Revenue
FROM customer_revenue_summary
ORDER BY Total_Revenue DESC
;

-- 2 Most Frequent Renters
 SELECT Customer_Name, Total_Rentals
 FROM customer_revenue_summary
 ORDER BY Total_Rentals DESC
 ;

-- 3 Customers Paying the Most Late Fees
SELECT Customer_Name, Total_Late_Fees
FROM customer_revenue_summary
ORDER BY Total_Late_Fees DESC
;

-- 4 Rental Frequency Segmentation
SELECT 
    CASE 
        WHEN Total_Rentals >= 20 THEN 'Frequent Renters'
        WHEN Total_Rentals BETWEEN 10 AND 19 THEN 'Regular Renters'
        ELSE 'Occasional Renters'
    END AS Rental_Frequency_Group,
    COUNT(Customer_ID) AS Customer_Count,
    ROUND(SUM(Total_Revenue), 2) AS Total_Revenue
FROM customer_revenue_summary
GROUP BY Rental_Frequency_Group
ORDER BY Total_Revenue DESC;

-- 5 Customers at Risk of Churn
SELECT Customer_Name, Last_Rental_Date
FROM customer_revenue_summary
HAVING Last_Rental_Date < DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
ORDER BY Last_Rental_Date ASC
;

-- 6 Revenue Breakdown by Most Watched Category
SELECT Most_Frequent_Category, 
       COUNT(Customer_ID) AS Customers_Who_Rent_This_Genre, 
       ROUND(SUM(Total_Revenue), 2) AS Total_Revenue
FROM customer_revenue_summary
GROUP BY Most_Frequent_Category
ORDER BY Total_Revenue DESC;

-- 7 Customer Lifetime Value
SELECT Customer_Name, 
       First_Rental_Date, 
       Last_Rental_Date, 
       ROUND(SUM(Total_Revenue), 2) AS Lifetime_Spending
FROM customer_revenue_summary
GROUP BY Customer_Name, First_Rental_Date, Last_Rental_Date
ORDER BY Lifetime_Spending DESC
;

-- 8 Customer Growth Over Time
SELECT DATE_FORMAT(First_Rental_Date, '%Y-%m') AS Customer_Cohort, 
       COUNT(Customer_ID) AS New_Customers, 
       ROUND(SUM(Total_Revenue), 2) AS Total_Revenue
FROM customer_revenue_summary
GROUP BY Customer_Cohort
ORDER BY Customer_Cohort;


-- 9 Revenue by Country
SELECT 
    Customer_Country AS Country, 
    COUNT(Customer_ID) AS Total_Customers, 
    ROUND(SUM(Total_Revenue), 2) AS Total_Revenue
FROM customer_revenue_summary
GROUP BY Customer_Country
ORDER BY Total_Revenue DESC;

-- 10 Revenue by City
SELECT 
    Customer_City AS City, 
    Customer_Country AS Country, 
    COUNT(Customer_ID) AS Total_Customers, 
    ROUND(SUM(Total_Revenue), 2) AS Total_Revenue
FROM customer_revenue_summary
GROUP BY Customer_City, Customer_Country
ORDER BY Total_Revenue DESC
;

-- 11 Average Spending Per Customer by Country
SELECT 
    Customer_Country AS Country, 
    COUNT(Customer_ID) AS Total_Customers, 
    ROUND(SUM(Total_Revenue), 2) AS Total_Revenue,
    ROUND(SUM(Total_Revenue) / COUNT(Customer_ID), 2) AS Avg_Revenue_Per_Customer
FROM customer_revenue_summary
GROUP BY Customer_Country
ORDER BY Avg_Revenue_Per_Customer DESC;
