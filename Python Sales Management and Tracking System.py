
# Retail Sales Management System
#
# Project Overview:
# This project demonstrates how Python fundamentals can be applied
# to a retail sales scenario. It collects product information,
# calculates sales totals, organizes transactions, applies discount
# rules, and handles basic data type errors.
#
# Objectives:
# - Collect product information and convert data types.
# - Organize transactions using tuples, dictionaries, and lists.
# - Apply business rules using loops and conditional statements.
# - Calculate discounts and final sales prices.
# - Create reusable functions and handle common errors.
#
# Technologies: Python 3, Jupyter Notebook
# Skills: Python Fundamentals, Data Structures, Business Logic,
# Sales Calculations, Basic Automation, Exception Handling


# ============================================================
# 1. USER INPUT AND DATA TYPE CONVERSION
# ============================================================

print("=" * 50)
print("RETAIL SALES MANAGEMENT SYSTEM")
print("=" * 50)

# Collect product information from the user
product_name = input("Enter the product name: ")
price_input = input("Enter the unit price: ")
quantity_input = input("Enter the quantity sold: ")

# Convert input values to the appropriate numeric types
price = float(price_input)
quantity = int(quantity_input)

# Calculate the total sales amount
total_sales = price * quantity

# Display the sales summary
print("\nSales Summary")
print("-" * 30)
print(f"Product: {product_name}")
print(f"Unit price: ${price:,.2f}")
print(f"Quantity sold: {quantity}")
print(f"Total sales: ${total_sales:,.2f}")


# ============================================================
# 2. DATA STRUCTURES FOR SALES RECORDS
# ============================================================

# Define the available product categories using a tuple
allowed_categories = ("Electronics", "Clothing", "Food")

print("\nAvailable Categories:", allowed_categories)

# Create a dictionary representing an individual transaction
specific_sale = {
    "product": product_name,
    "category": allowed_categories[0],
    "price": price,
    "quantity": quantity,
    "total": total_sales
}

# Create a list to store sales transactions
sales_register = []
sales_register.append(specific_sale)

print("\nSales Register:")
for sale in sales_register:
    print(sale)


# ============================================================
# 3. CONTROL FLOW AND DISCOUNT RULES
# ============================================================

# Create sample transactions for testing
sale_2 = {
    "product": "Sweatshirt",
    "category": "Clothing",
    "price": 70000.0,
    "quantity": 3,
    "total": 210000.0
}

sale_3 = {
    "product": "Hamburger",
    "category": "Food",
    "price": 15000.0,
    "quantity": 5,
    "total": 75000.0
}

# Add sample transactions to the sales register
sales_register.append(sale_2)
sales_register.append(sale_3)

# Process transactions and apply a 10% discount
# when the original total exceeds $100,000
print("\nSales Report")
print("=" * 50)

for sale in sales_register:
    original_total = sale["total"]

    if original_total > 100000:
        discount = original_total * 0.10
        final_total = original_total - discount

        print(f"\nProduct: {sale['product']}")
        print(f"Original total: ${original_total:,.2f}")
        print(f"Discount applied (10%): ${discount:,.2f}")
        print(f"Final total: ${final_total:,.2f}")
    else:
        print(f"\nProduct: {sale['product']}")
        print(f"Original total: ${original_total:,.2f}")
        print("No discount applied.")

    print("-" * 50)


# ============================================================
# 4. FUNCTIONS AND EXCEPTION HANDLING
# ============================================================

# Define a reusable function to calculate sales revenue
def calculate_revenue(price, quantity):
    try:
        revenue = price * quantity
        return revenue
    except TypeError:
        print("Error: Price and quantity must be numeric values.")
        return 0


# Test 1: Valid numeric inputs
result_1 = calculate_revenue(1000, 3.5)
print(f"\nTest 1 - Revenue: ${result_1:,.2f}")

# Test 2: Invalid input to trigger exception handling
result_2 = calculate_revenue("one thousand", 3.5)
print(f"Test 2 - Revenue: ${result_2:,.2f}")


# ============================================================
# PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("PROJECT EXECUTION COMPLETED")
print(f"Total transactions processed: {len(sales_register)}")
print("=" * 50)