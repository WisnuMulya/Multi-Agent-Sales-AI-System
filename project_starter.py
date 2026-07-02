import json

import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from dataclasses import dataclass, asdict, field
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE LOWER(item_name) = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name.lower(), "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIServerModel(
    model_id="gpt-5-mini",
    api_key=openai_api_key,
)


# Initialize running state
@dataclass
class ItemOrder:
    item_name: str
    requested_quantity: int
    supplier_price_per_unit: float = 0.0
    retail_price_per_unit: float = 0.0
    current_stock: int = 0
    min_stock_level: int = 0
    stock_order_needed: int = 0
    estimated_delivery_date: str = "Immediate"
    is_stock_ordered: bool = False
    is_fulfilled: bool = False

@dataclass
class OrderRunningState:
    requested_items: List[ItemOrder] = field(default_factory=list)
    as_of_date: str = ""
    cash_balance: float = 0.0
    inventory_value: float = 0.0
    total_assets: float = 0.0
    customer_job: str = ""
    need_size: str = ""
    event: str = ""
    initial_request: str = ""

    def reset(self):
        self.requested_items.clear()
        self.as_of_date = ""
        self.cash_balance = 0.0
        self.inventory_value = 0.0
        self.total_assets = 0.0
        self.customer_job = ""
        self.need_size = ""
        self.event = ""

order_running_state = OrderRunningState()

# Tools for inventory agent

def get_minimum_stock_level(item_name: str) -> int:
    """
    Retrieves the minimum stock level for a given item.

    Args:
        item_name (str): The name of the item to check.

    Returns:
        int: The minimum stock level for the specified item.
    """
    item_name = item_name.lower()
    
    # Query the inventory table for the minimum stock level of the specified item
    query = "SELECT min_stock_level FROM inventory WHERE LOWER(item_name) = :item_name"
    result = pd.read_sql(query, db_engine, params={"item_name": item_name})

    if not result.empty:
        return int(result.iloc[0]["min_stock_level"])
    else:
        return 0  # Return 0 if the item is not found in the inventory


@tool
def get_item_stocks() -> Dict[str, Union[Dict[str, Union[int, str, bool]], str]]:
    """
    Retrieves the available stock levels for a list of requested items for the current order.

    Returns:
        Dict[str, Union[Dict[str, Union[int, str, bool]], str]]: A dictionary information for each item.

    Example:
        >> get_item_stocks()
        {
            "A4 paper": {"current_stock": 500, "min_stock_level": 100, "requested_quantity": 450, "stock_order_needed": 250, "estimated_delivery_date": "2025-01-05"},
            "Letter-sized paper": {"current_stock": 300, "min_stock_level": 80, "requested_quantity": 250, "stock_order_needed": 190, "estimated_delivery_date": "2025-01-02"},
        }
    """
    result = {"available_items": []}

    # Get all requested items from the running state
    requested_items = order_running_state.requested_items
    as_of_date = order_running_state.as_of_date
    inventory_snapshot = get_all_inventory(as_of_date)
    inventory_item_names = {item_name.lower() for item_name in inventory_snapshot.keys()}
    valid_item_names = {item["item_name"].lower() for item in paper_supplies}.union(inventory_item_names)

    for item in requested_items:
        if not isinstance(item, ItemOrder):
            print(f"WARN: Invalid item in requested_items: {item}. Skipping.")
            continue

        item_name = item.item_name.lower()
        if item_name not in valid_item_names:
            print(f"WARN: Item '{item_name}' not found in valid item names. Skipping.")
            continue

        requested_quantity = item.requested_quantity
        current_stock = get_stock_level(item_name, as_of_date)["current_stock"].iloc[0]
        item.current_stock = int(current_stock)
        min_stock_level = get_minimum_stock_level(item_name)
        item.min_stock_level = int(min_stock_level)

        # Calculate if a stock order is needed, accounting for the minimum stock level
        stock_order_needed = max(0, requested_quantity - current_stock + min_stock_level)
        item.stock_order_needed = int(stock_order_needed)

        # Get estimated delivery date based on stock order needed
        estimated_delivery_date = get_supplier_delivery_date(as_of_date, stock_order_needed) if stock_order_needed > 0 else "Immediate"
        item.estimated_delivery_date = estimated_delivery_date

        result['available_items'].append({
            "item_name": item_name,
            "current_stock": int(current_stock),
            "min_stock_level": int(min_stock_level),
            "requested_quantity": int(requested_quantity),
            "stock_order_needed": int(stock_order_needed),
            "estimated_delivery_date": estimated_delivery_date,
        })

    if not result.get('available_items'):
        return {"message": "No available items for the requested order."}

    return result

@tool
def get_inventory_snapshot(as_of_date: str) -> List[Dict[str, Union[str, int, float]]]:
    """
    Retrieve a snapshot of the inventory as of a specific date.

    Args:
        as_of_date (str): The date (inclusive) for which to retrieve the inventory snapshot in ISO format.

    Returns:
        List[Dict[str, Union[str, int, float]]]: A list of dictionaries representing each item in the inventory,
        including item name, category, unit price, current stock, and minimum stock level.

    Raises:
        ValueError: If no inventory snapshot is found for the specified date.
    """
    # Query the inventory table for all items
    financial_report = generate_financial_report(as_of_date)
    inventory_snapshot = financial_report.get("inventory_summary", [])

    if not inventory_snapshot:
        raise ValueError(f"No inventory snapshot found for the date: {as_of_date}")
    else:
        return inventory_snapshot

# Tools for quoting agent
@tool
def get_ordered_items_prices() -> Dict[str, float]:
    """
    Retrieve the unit prices for requested items with a profit margin of 20% applied.

    Returns:
        Dict[str, float]: A dictionary mapping item names to their unit prices.
    """
    prices_with_margin = {}

    for item in order_running_state.requested_items:
        if not isinstance(item, ItemOrder):
            print(f"WARN: Invalid item in requested_items: {item}. Skipping.")
            continue

        prices_with_margin[item.item_name] = item.retail_price_per_unit

    if not prices_with_margin:
        return {"message": "No available items for the requested order."}

    return prices_with_margin

@tool
def search_quote_history_based_on_order() -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided order information.

    The function searches both the original customer request (from `quote_requests`) and
    the response quote (from `quotes`) for each item-name in the order-request.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
        
    Example:
        >> search_quote_history_based_on_order(limit=3)
        [
            {
                "original_request": "Customer requested A4 paper and Banner paper for an event.",
                "total_amount": 150.00,
                "quote_explanation": "Based on previous orders, we can offer a discount for bulk purchase.",
                "job_type": "Event",
                "order_size": "Large",
                "event_type": "Corporate",
                "order_date": "2025-01-10"
            },
            ...
    """
    # Extract item names from the order information
    order_info = order_running_state.requested_items
    search_terms = [item.item_name for item in order_info if isinstance(item, ItemOrder) and item.item_name]

    # Use the existing search_quote_history function to find matching quotes
    result = search_quote_history(search_terms)
    if not result:
        result = [{"message": "No matching quotes found for the requested items."}]

    return result


# Tools for ordering agent

@tool
def process_all_orders() -> Dict[str, Union[float, List[Dict[str, Union[str, int, float]]]]]:
    """
    Process the orders for all requested items, placing stock orders if needed and finalizing sales transactions.

    Returns:
        Dict[str, Union[float, List[Dict[str, Union[str, int, float]]]]]: A dictionary containing the total sales amount,
        total stock order amount, and a list of processed orders.
        The processed orders list includes dictionaries with keys: item_name, quantity_ordered, price_per_unit, total_price, date, and transaction_type.

    Raises:
        ValueError: If an error occurs during order processing.
    """
    stock_orders = []
    sales_transactions = []
    as_of_date = order_running_state.as_of_date
    total_sales_amount = 0.0
    total_stock_order_amount = 0.0

    for item in order_running_state.requested_items:
        if not isinstance(item, ItemOrder):
            print(f"WARN: Invalid item in requested_items: {item}. Skipping.")
            continue

        # Place stock order if needed
        if item.stock_order_needed > 0 and not item.is_stock_ordered:
            try:
                transaction_id = create_transaction(
                    item_name=item.item_name,
                    transaction_type="stock_orders",
                    quantity=item.stock_order_needed,
                    price=item.stock_order_needed * item.supplier_price_per_unit,
                    date=as_of_date
                )
                item.is_stock_ordered = True
                total_stock_order_amount += item.stock_order_needed * item.supplier_price_per_unit
                print(f"Stock order placed for {item.item_name}: {item.stock_order_needed} units (Transaction ID: {transaction_id})")
            except Exception as e:
                raise ValueError(f"Failed to place stock order for {item.item_name}: {e}")
            
            stock_orders.append({
                "item_name": item.item_name,
                "quantity_ordered": item.stock_order_needed,
                "price_per_unit": item.supplier_price_per_unit,
                "total_price": item.stock_order_needed * item.supplier_price_per_unit,
                "transaction_type": "stock_orders",
                "date": as_of_date
            })

        # Finalize sales transaction
        try:
            transaction_id = create_transaction(
                item_name=item.item_name,
                transaction_type="sales",
                quantity=item.requested_quantity,
                price=item.requested_quantity * item.retail_price_per_unit,
                date=as_of_date
            )
            item.is_fulfilled = True
            total_sales_amount += item.requested_quantity * item.retail_price_per_unit
            print(f"Sales transaction completed for {item.item_name}: {item.requested_quantity} units (Transaction ID: {transaction_id})")
        except Exception as e:
            raise ValueError(f"Failed to finalize sales transaction for {item.item_name}: {e}")

        sales_transactions.append({
            "item_name": item.item_name,
            "quantity_ordered": item.requested_quantity,
            "price_per_unit": item.retail_price_per_unit,
            "total_price": item.requested_quantity * item.retail_price_per_unit,
            "transaction_type": "sales",
            "date": as_of_date
        })

    result = {
        "total_sales_amount": round(total_sales_amount, 2),
        "total_stock_order_amount": round(total_stock_order_amount, 2),
        "stock_orders": stock_orders,
        "sales_transactions": sales_transactions
    }

    return result

@tool
def get_cash_balance_as_of_date(as_of_date: str) -> float:
    """
    Retrieve the current cash balance as of a specific date.

    Args:
        as_of_date (str): The date (inclusive) for which to retrieve the cash balance in ISO format.

    Returns:
        float: The current cash balance as of the specified date.
    """
    return get_cash_balance(as_of_date)


# Set up your agents and create an orchestration agent that will manage them.

class CustomerRequestAgent(ToolCallingAgent):
    """
    Agent responsible for parsing customer requests and extracting structured order information.
    """

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="customer_request_agent",
            description="Agent that parses customer requests and extracts structured order information."
        )

class InventoryManagementAgent(ToolCallingAgent):
    """
    Agent responsible for checking inventory levels, determining if items are
    in stock, determining stock replenishment orders, and estimating delivery times.
    """

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[get_item_stocks, get_inventory_snapshot],
            model=model,
            name="inventory_management_agent",
            description="Agent that manages inventory levels and determines if stock orders are needed to fulfill customer requests."
        )


class QuotingAgent(ToolCallingAgent):
    """
    Agent responsible for generating quotes based on customer requests and 
    historical quote data, ensuring that quotes are persuasive, competitive,
    and profitable.
    """

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[search_quote_history_based_on_order, get_ordered_items_prices],
            model=model,
            name="quoting_agent",
            description="Agent that generates quotes based on customer requests and historical data."
        )


class OrderingAgent(ToolCallingAgent):
    """
    Agent responsible for placing orders to suppliers and finalizing sales 
    transactions, ensuring timely delivery and accurate record-keeping.
    """

    def __init__(self, model:OpenAIServerModel):
        super().__init__(
            tools=[process_all_orders, get_cash_balance_as_of_date],
            model=model,
            name="ordering_agent",
            description="Agent that places orders to suppliers and finalizes sales transactions."
        )


class OrchestrationAgent(ToolCallingAgent):
    """
    Orchestration agent that coordinates the activities of the customer-request,
    inventory-management, quoting, and ordering agents to handle customer requests.
    """

    def __init__(self, model: OpenAIServerModel):
        self.customer_request_agent = CustomerRequestAgent(model)
        self.inventory_management_agent = InventoryManagementAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.ordering_agent = OrderingAgent(model)

        @tool
        def process_initial_request() -> str:
            """
            Parse the initial customer request to extract structured order information.
            You are not able to ask confirmations to customers.

            Returns:
                str: A structured summary of the parsed order information.

            Raises:
                ValueError: If the response from the customer request agent cannot be parsed as valid JSON.
            """
            canon_item_names = [{"item_name": item["item_name"], "price_per_unit": item.get("unit_price", 0.0)} for item in paper_supplies]
            input = f"""
            # Role & Task
            You are a customer-request parsing agent.
            Your task is to parse the customer request and extract structured order information.
            The requested item-names must be replaced with the canon item-names from the list below.

            # Suggested Workflow
            1. Map the requested item names to the canon item-names from the list below, ensuring that the downstream agents can recognize them. 
            2. If an item cannot be mapped to the list, you must put it in the "unavailable_items" list.
            3. Format the extracted information into a structured summary.

            # Output Format in JSON
            {{
                "requested_items": [
                    {{
                        "item_name" (str): <Canon Item Name>,
                        "quantity" (int): <Quantity>
                        "supplier_price_per_unit" (float): <Supplier Price Per Unit>
                    }},
                    ...
                ]
                "unavailable_items": ["<Item Name>", ...],
                "request_date" (str): <Request Date in ISO format (YYYY-MM-DD)>
            }}

            # Customer Order
            {order_running_state.initial_request}
            Request Date: {order_running_state.as_of_date}

            # Canon Item Names
            {canon_item_names}
            """
            response = self.customer_request_agent.run(input)
            canon_item_names_lower = {item["item_name"].lower() for item in canon_item_names}

            try:
                # Attempt to parse the response as JSON
                parsed_response = json.loads(response)
                for item_order in parsed_response.get("requested_items", []):
                    item_name = item_order.get("item_name", "")

                    if item_name.lower() not in canon_item_names_lower or not item_name:
                        print(f"WARN: Item '{item_name}' not found in canon item names. Skipping.")
                        continue  # Skip items not in the canon list

                    quantity = item_order.get("quantity", 0)

                    # Get price per unit from the parsed item, otherwise get from paper_supplies
                    if "supplier_price_per_unit" not in item_order:
                        # Look up price from paper_supplies if not provided in the order
                        supplier_price_per_unit = next((item["unit_price"] for item in paper_supplies if item["item_name"].lower() == item_name.lower()), 0.0)
                    elif "supplier_price_per_unit" in item_order:
                        supplier_price_per_unit = item_order.get("supplier_price_per_unit")
                    else:
                        continue  # Skip if supplier_price_per_unit is not provided and cannot be found

                    # Add 20% profit margin to the price per unit
                    retail_price_per_unit = supplier_price_per_unit * 1.2
                    retail_price_per_unit = round(retail_price_per_unit, 2)  # Round to 2 decimal places
                    supplier_price_per_unit = round(supplier_price_per_unit, 2)  # Round to 2 decimal places

                    if item_name and quantity > 0:
                        order_running_state.requested_items.append(
                            ItemOrder(item_name=item_name, requested_quantity=quantity, supplier_price_per_unit=supplier_price_per_unit, retail_price_per_unit=retail_price_per_unit)
                        )
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON response from process_customer_order: {response}. Error: {e}")
            
            order = asdict(order_running_state)
            result = {"requested_items": [], "request_date": order_running_state.as_of_date, "unavailable_items": parsed_response.get("unavailable_items", [])}
            for item in order["requested_items"]:
                result["requested_items"].append({
                    "item_name": item['item_name'],
                    "quantity": item['requested_quantity'],
                    "retail_price_per_unit": item['retail_price_per_unit'],
                })

            return result
        
        @tool
        def process_inventory_check() -> str:
            """
            Check inventory availability and levels based on the parsed order information and determine if stock replenishment is needed.

            Returns:
                str: A structured summary of the inventory check results.
            """
            # Use the inventory management agent to check inventory
            ordered_items = asdict(order_running_state)["requested_items"]
            ordered_items_info = [{"item_name": item["item_name"], "requested_quantity": item["requested_quantity"]} for item in ordered_items]
            input = f"""
            # Role & Task
            You are an inventory management agent.
            Your task is to check inventory availability and levels, and to determine if stock replenishment is needed based on the parsed order information.
            Do not call get_inventory_snapshot for order processing, as it is not needed.

            # Workflow for Order Processing
            1. Call get_item_stocks to check requested item stocks.
            2. Based on the output of get_item_stocks, determine if any stock replenishment is needed.
            3. Based on the output of get_item_stocks, provide an estimated delivery date for any stock replenishment needed.
            4. Do not include any items other than the available ones.
            5. Format the output into a structured summary.

            # Output Format in JSON
            [
                {{
                    "item_name" (str): The canon item name from the inventory or the requested item name if invalid,
                    "requested_quantity" (int): The quantity requested by the customer,
                    "stock_order_needed" (int): The quantity that needs to be ordered from the supplier,
                    "estimated_delivery_date" (str): The estimated delivery date for the stock order if needed, or "Immediate" if no stock order is needed,
                }}
            ]

            # Request Information
            {ordered_items_info}
            Request Date: {order_running_state.as_of_date}
            """
            response = self.inventory_management_agent.run(input)

            return response
        
        
        @tool
        def process_quote() -> str:
            """
            Generate a quote based on the order information, historical quote data, and supplier prices.

            Returns:
                str: A structured summary of the generated quote.
            """
            ordered_items = asdict(order_running_state)["requested_items"]
            ordered_items_info = [{"item_name": item["item_name"], "requested_quantity": item["requested_quantity"], "estimated_delivery_date": item.get("estimated_delivery_date", "Immediate")} for item in ordered_items]
            input = f"""
            # Role & Task
            You are a quoting agent of the Munder Difflin Paper Company.
            Your task is to generate a quote based on the requested order information and historical quote data, if possible.
            Do not inform the customer about anything to do with the stock's shortages, stock orders, or our profit margin.
            Do not mention any internal processes or tools used and do not ask for confirmation from the customer.

            # Suggested Workflow
            1. First, get_ordered_items_prices to retrieve the unit prices for all items in the current order.
            2. Then, use search_quote_history_based_on_order to find relevant historical quotes that match the order information.
            3. Based on the historical quotes and the current order information, if any, generate a new quote that is persuasive.
            4. Write a quote that is similar to the style of the historical quote-explanations, but do not copy them verbatim and no placeholders.

            # Order Information
            Request Date: {order_running_state.as_of_date}
            Ordered Items: {ordered_items_info}
            """
            response = self.quoting_agent.run(input)
            order_running_state.quote_text = response  # Store the quote text in the running state

            return response
        
        @tool
        def process_ordering() -> str:
            """
            Finalize the order by recording the sales transaction and placing stock orders.

            Returns:
                str: A structured summary of the ordering process, including any stock orders placed and the sales transaction recorded.
            """
            input = f"""
            # Role & Task
            You are an ordering agent.
            Your task is to finalize the requested order by recording the sales transaction and placing stock orders.
            Do not inform about our cash information. Only say whether the order can be fulfilled or not.

            # Suggested Workflow
            1. Call process_orders to record the sales transaction and place stock orders for any items that need replenishment.
            2. Call get_cash_balance_as_of_date to retrieve the current cash balance after processing the orders.
            3. Calculate if we have sufficient cash to cover the stock orders, if not, return a final answer saying the order cannot be fulfilled due to insufficient cash.
            4. Format the output into a structured summary of the ordering process.

            # Request Information
            Request Date: {order_running_state.as_of_date}

            # Output Format in JSON
            {{
                "sales_transactions_recorded" (List[str]): A list of items that were successfully recorded as sales transactions,
                "notes" (str): Whether all orders can be fulfilled, based on our cash balance.
            }}
            """
            response = self.ordering_agent.run(input)
            return response

        super().__init__(
            tools=[process_initial_request, process_inventory_check, process_quote, process_ordering],
            model=model,
            name="orchestration_agent",
            description="""
            # Role & Task
            You are an orchestration agent for Munder Difflin Paper Company.
            You coordinate the activities of the customer-request, inventory-management, quoting, and ordering agents to handle customer requests.
            Do not inform the customer about anything to do with the stock orders, our profit margin, or our cash balance. And do not mention any internal processes or tools used.
            Assume all stock orders will be fulfilled and delivered on time. Do not follow up on stock orders.
            Do not call any tools to ask for confirmation from the customer. You are the one who must do that in your final answer.
            For the final answer, you must use the generated quote text from the quoting-agent as the final answer to the customer with modification about unavailable items, delays, or asking confirmation.

            # Workflow
            1. Use the `process_initial_request` tool to extract and validate the customer's request information.
            2. After processing the request, you must use the `process_inventory_check` tool to check the processed request for availability and estimated delivery times.
            3. After obtaining the availability and delivery times information, you must use the `process_quote` tool to generate a quote.
            4. After obtaining the quote, you must use the `process_ordering` tool to finalize all orders.
            5. After finalizing the orders, you must return a final answer to the customer using the `final_answer` tool.
            """
        )

    def handle_customer_request(self, customer_job: str, need_size: str, event: str, request: str, request_date: str) -> str:
        """
        Handle a customer request through a coordinated multi-agent workflow.
        """
        order_running_state.reset()
        order_running_state.customer_job = customer_job
        order_running_state.need_size = need_size
        order_running_state.event = event
        order_running_state.as_of_date = request_date
        order_running_state.initial_request = request

        input = f"""
        # Context
        Customer Job: {customer_job}
        Need Size: {need_size}
        Event: {event}
        Request: {request}
        Request Date: {request_date}
        """

        return self.run(input)

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return
    print("Database initialized and test data loaded successfully.")

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]
    orchestration_agent = OrchestrationAgent(model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        response = orchestration_agent.handle_customer_request(
            customer_job=row["job"],
            need_size=row["need_size"],
            event=row["event"],
            request=row["request"],
            request_date=request_date
        )

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": round(current_cash, 2),
                "inventory_value": round(current_inventory, 2),
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
