
# ============================================================
# PROJECT: SAKILA SQL BUSINESS ANALYSIS
# Database: MySQL Sakila Sample Database
# ============================================================

USE sakila;


# ============================================================
# PART 1: SELECT AND WHERE
# ============================================================

# 1. Retrieve the first and last names of all customers.

SELECT first_name, last_name
FROM customer;


# 2. Retrieve films longer than 120 minutes.

SELECT title, length
FROM film
WHERE length > 120;


# ============================================================
# PART 2: ORDER BY AND LIMIT
# ============================================================

# 3. Sort customers alphabetically by last name.

SELECT first_name, last_name
FROM customer
ORDER BY last_name ASC;


# 4. Retrieve the five longest films.

SELECT title, length
FROM film
ORDER BY length DESC
LIMIT 5;


# ============================================================
# PART 3: INNER JOIN
# ============================================================

# 5. Retrieve customer names and their payment information.

SELECT
    c.first_name,
    c.last_name,
    p.amount,
    p.payment_date
FROM payment AS p
INNER JOIN customer AS c
    ON p.customer_id = c.customer_id;


# 6. Retrieve rented film titles and rental dates.

SELECT
    f.title,
    r.rental_date
FROM rental AS r
INNER JOIN inventory AS i
    ON r.inventory_id = i.inventory_id
INNER JOIN film AS f
    ON i.film_id = f.film_id;


# ============================================================
# PART 4: LEFT JOIN
# ============================================================

# 7. Find customers who have never made a payment.

SELECT
    c.first_name,
    c.last_name
FROM customer AS c
LEFT JOIN payment AS p
    ON c.customer_id = p.customer_id
WHERE p.payment_id IS NULL;


# 8. Find films without associated actors.

SELECT
    f.title,
    f.length
FROM film AS f
LEFT JOIN film_actor AS fa
    ON f.film_id = fa.film_id
WHERE fa.film_id IS NULL;


# ============================================================
# PART 5: INSERT, UPDATE, AND DELETE
# ============================================================

# WARNING: These statements modify database records.
# Execute them only in a test environment.


# 9. Insert a temporary actor if the record does not exist.

INSERT INTO actor (first_name, last_name)
SELECT 'Diego', 'Temporary'
WHERE NOT EXISTS (
    SELECT 1
    FROM actor
    WHERE first_name = 'Diego'
      AND last_name = 'Temporary'
);


# 10. Update the temporary actor's first name.

UPDATE actor
SET first_name = 'DiegoUpdated'
WHERE first_name = 'Diego'
  AND last_name = 'Temporary';


# 11. Delete the temporary actor.

DELETE FROM actor
WHERE first_name = 'DiegoUpdated'
  AND last_name = 'Temporary';


# ============================================================
# PART 6: ADVANCED QUERIES AND BUSINESS ANALYSIS
# ============================================================

# 12. Find the five customers with the highest total payments.

SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    SUM(p.amount) AS total_paid
FROM customer AS c
INNER JOIN payment AS p
    ON c.customer_id = p.customer_id
GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name
ORDER BY total_paid DESC
LIMIT 5;


# 13. Find the five most frequently rented films.

SELECT
    f.film_id,
    f.title,
    COUNT(r.rental_id) AS rental_count
FROM rental AS r
INNER JOIN inventory AS i
    ON r.inventory_id = i.inventory_id
INNER JOIN film AS f
    ON i.film_id = f.film_id
GROUP BY
    f.film_id,
    f.title
ORDER BY rental_count DESC
LIMIT 5;


# ============================================================
# END OF PROJECT
# ============================================================
