import os
from time import strftime
from apscheduler.schedulers.blocking import BlockingScheduler
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
import network_utils
import model_utils
from helpers import apology, login_required, lookup, usd
from django.shortcuts import render
from plotly.offline import plot
from plotly.graph_objs import Scatter
# Configure application
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///stockbot.db")


@app.template_filter()
def currencyFormat(value):
    value = float(value)
    return "${:,.2f}".format(value)


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response




@app.route("/")
def index():
    """Show portfolio of stocks and graphs"""
    # update stock values
    update_stocks()
    # get currently owned stocks
    portfolio, stocks = get_owned_stocks()
    print(portfolio, stocks)
    # stocks = stocks
    # print(stocks)


    # print(portfolio)
    return render_template(
        "index.html",
        port_data=portfolio,
        stock_data = stocks
    )


@app.route("/buy", methods=["GET", "POST"])
def request_stock():
    if request.method == "POST":
        print(request.form)
        days = int(request.form.get('symbol'))
        for i in range(days):
            make_predict(days-i)
        return redirect("/")
    return render_template("buy.html")


@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    transactions = db.execute(
        """SELECT price, symbol, timestamp, amount FROM transactions WHERE userid = ?""",
        session["user_id"],
    )
    return render_template("history.html", transactions=transactions)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute(
            "SELECT * FROM users WHERE username = ?", request.form.get("username")
        )

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        if symbol:
            value = lookup(symbol)
            if value:
                return render_template(
                    "qoute-value.html",
                    value=str(usd(value["price"])),
                    symbol=value["name"],
                )

        return apology("Invalid Symbol", 400)
    else:
        return render_template("qoute.html")


@app.route("/add_cash", methods=["GET", "POST"])
@login_required
def add_cash():
    """Get stock quote."""
    if request.method == "POST":
        amount = request.form.get("cash")
        try:
            amount = int(amount)
            if (
                db.execute(
                    """SELECT cash FROM users WHERE id = ?""", session["user_id"]
                )[0]["cash"]
                + amount
                > 9223372036854775806
            ):
                raise ValueError
            db.execute(
                """UPDATE users
                       SET cash = ?
                       WHERE id = ?""",
                db.execute(
                    """SELECT cash FROM users WHERE id = ?""", session["user_id"]
                )[0]["cash"]
                + amount,
                session["user_id"],
            )

        except ValueError:
            return apology("Invalid Amount of Cash", 400)
        return redirect("/")

    else:
        return render_template("add_cash.html")


# CREATE TABLE transactions(transactionid INTEGER PRIMARY KEY AUTOINCREMENT,
#                            userid INTEGER,
#                            price DECIMAL(10,2) NOT NULL,
#                            symbol VARCHAR(10) NOT NULL,
# timestamp NOT NULL DEFAULT(GETDATE()),
# amount INTEGER NOT NULL,
# FOREIGN KEY (userid) REFERENCES users(id));

# CREATE TABLE stocks(stockid INTEGER PRIMARY KEY AUTOINCREMENT,
# price DECIMAL(10, 2) NOT NULL,
# symbol VARCHAR(10) UNIQUE NOT NULL);
# CREATE TABLE holdings(stockholdingid INTEGER PRIMARY KEY AUTOINCREMENT,
# userid INTEGER,
# stockid INTEGER,
# amount INTEGER,
#  FOREIGN KEY (userid) REFERENCES users(id),
# #  FOREIGN KEY (stockid) REFERENCES stocks(id));
# {% comment %} <header>
#     <div class="alert alert-primary mb-0 text-center" role="alert">
#         Registered!
#     </div>
# </header> {% endcomment %}


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 400)
        elif (request.form.get("password") != request.form.get("confirmation")):
            return apology("Password is not the same as confirmation", 400)
        # Query database for username
        rows = db.execute(
            "SELECT * FROM users WHERE username = ?", request.form.get("username")
        )
        # check if it already exists
        if len(rows) > 0:
            return apology("Username already in use", 400)

        db.execute(
            """INSERT INTO users (username, hash)
                   VALUES (?, ?)""",
            request.form.get("username"),
            generate_password_hash(request.form.get("password")),
        )
        return render_template("login.html")

    else:
        return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""
    user_stocks = get_owned_stocks(session["user_id"])
    if request.method == "POST":
        try:
            symbol = request.form.get("symbol")
            amount = int(request.form.get("shares"))
            if not symbol in user_stocks:
                return apology("Stock not owned", 400)
            if amount > user_stocks[symbol][1]:
                return apology("Too many stocks sold", 400)
            db.execute(
                """INSERT INTO transactions (userid, price, symbol, timestamp, amount) VALUES (?, ?, ?, ?, ?)""",
                session["user_id"],
                user_stocks[symbol][0],
                symbol,
                db.execute("""SELECT CURRENT_TIMESTAMP""")[0]["CURRENT_TIMESTAMP"],
                (0 - amount),
            )
            difference = user_stocks[symbol][1] - amount
            if difference == 0:
                db.execute(
                    """DELETE FROM holdings
                           WHERE stockholdingid = ?""",
                    user_stocks[symbol][3],
                )
            else:
                db.execute(
                    """UPDATE holdings
                           SET amount = ?
                           WHERE stockholdingid = ?""",
                    difference,
                    user_stocks[symbol][3],
                )
            db.execute(
                """UPDATE users
                       SET cash = ?
                       WHERE id = ?""",
                db.execute(
                    """SELECT cash FROM users WHERE id = ?""", session["user_id"]
                )[0]["cash"]
                + user_stocks[symbol][0] * amount,
                session["user_id"],
            )
        except ValueError:
            return apology("Invalid Number", 415)
        return redirect("/")
    # print(list(user_stocks.keys()))
    return render_template("sell.html", result=list(user_stocks.keys()))

def get_amount_owned(symbol):
    stocks = db.execute("""SELECT * FROM holdings where symbol = ?""", symbol)
    total = 0
    for i in stocks:
        total+=i['amount']
    return total
def get_amount_owned_regress(symbol):
    stocks = db.execute("""SELECT * FROM modelHoldings where symbol = ?""", symbol)
    total = 0
    for i in stocks:
        total+=i['amount']
    return total

def get_balance(symbol,regress=False):
    balance = 10000
    if regress:
        stock_transactions = db.execute("""SELECT * FROM modelACTIONS WHERE symbol = ? ORDER BY timestampe""",symbol)
    else:
        stock_transactions = db.execute("""SELECT * FROM ACTIONS WHERE symbol = ? ORDER BY timestampe""",symbol)
    for trans in stock_transactions: 
            if trans['action'] == 1: # buy
                balance -= trans['amount']*trans['PRICE']
            elif trans['action'] == 2: # sell 
                balance += trans['amount']*trans['PRICE']
    return balance

def get_owned_stocks():
    symbols = set([i['symbol'] for i in db.execute("""SELECT symbol FROM ACTIONS""")]) # should be the same symbol(s)
    graph_data = []
    stocks_data = []
    for symbol in symbols:
        balance = 10000
        stock_transactions = db.execute("""SELECT * FROM ACTIONS WHERE symbol = ? ORDER BY timestampe""",symbol) # f*ck, I can't belive I misspelled timetamp.
        lables = [i["timestampe"] for i in stock_transactions]# This is what I get for coding at 3 am.
        # data = [i["amount"]*i["PRICE"] for i in stock_transactions]
        amount_held = 0
        data = []
        stock_data = []
        actions = []
        # print(symbol)
        for trans in stock_transactions: # wtf is this shit, TODO: write non-shit code
            if trans['action'] == 1: # buy
                amount_held += trans['amount']
                balance -= trans['amount']*trans['PRICE']
            elif trans['action'] == 2: # sell 
                amount_held -= trans['amount']
                balance += trans['amount']*trans['PRICE']
            stock_value = amount_held*trans['PRICE']
            data.append(balance+stock_value)
            stock_data.append(trans['PRICE'])
            actions.append(trans['action'])
        stocks_data.append((symbol+" stock value",lables, stock_data))
        graph_data.append((symbol + " network holding value", lables, data, actions))
        # Oh, you thought that was bad?! Just wait, it gets worse!
        balance = 10000
        stock_transactions = db.execute("""SELECT * FROM modelACTIONS WHERE symbol = ? ORDER BY timestampe""",symbol) # f*ck, I can't belive I misspelled timetamp.
        lables = [i["timestampe"] for i in stock_transactions]# This is what I get for coding at 3 am.
        # data = [i["amount"]*i["PRICE"] for i in stock_transactions]
        amount_held = 0
        data = []
        stock_data = []
        actions = []
        for transact in stock_transactions:
            if transact['action'] == 1: # buy
                amount_held += transact['amount']
                balance -= transact['amount']*transact['PRICE']
            elif transact['action'] == 2: # sell 
                amount_held -= transact['amount']
                balance += transact['amount']*transact['PRICE']
            stock_value = amount_held*transact['PRICE']
            data.append(balance+stock_value)
            actions.append(transact['action'])
        graph_data.append((symbol + "regression model holding value", lables, data, actions))
    print(graph_data, stock_data)
    return graph_data, stocks_data

def update_evolved():
    symbols = len([i['symbol'] for i in db.execute("SELECT DISTINCT symbol from ACTIONS")])
    print(symbols)
    stock = [i['symbol'] for i in db.execute("SELECT DISTINCT symbol FROM (SELECT symbol, timestampe from ACTIONS ORDER BY timestampe DESC LIMIT ?) WHERE timestampe != DATE('now');",symbols)]
    amount = 10
    print(stock)
    for symbol in stock:
        print(symbol)
        action = network_utils.get_action(symbol,0,get_amount_owned(symbol),get_balance(symbol))
        db.execute("INSERT INTO ACTIONS (price, symbol, timestampe, action, amount) VALUES (?, ?, ?, ?, ?)",
                    network_utils.get_current_price(symbol),
                    symbol,
                    db.execute("""SELECT DATE('now')""")[0]["DATE('now')"],
                    action,
                    amount)
        
def update_regression():
    symbols = len([i['symbol'] for i in db.execute("SELECT DISTINCT symbol from modelACTIONS")])
    stock = [i['symbol'] for i in db.execute("SELECT DISTINCT symbol FROM (SELECT symbol, timestampe from modelACTIONS ORDER BY timestampe DESC LIMIT ?) WHERE timestampe != DATE('now');",symbols)]
    amount = 10
    print(symbols)
    print(stock)
    for symbol in stock:
        print(symbol)
        action = model_utils.get_action(symbol,0,get_amount_owned_regress(symbol),balance=get_balance(symbol, True))
        print("Action " + str(action))
        db.execute("INSERT INTO modelACTIONS (price, symbol, timestampe, action, amount) VALUES (?, ?, ?, ?, ?)",
                    model_utils.get_current_price(symbol),
                    symbol,
                    db.execute("""SELECT DATE('now')""")[0]["DATE('now')"],
                    action,
                    amount)
def update_stocks():
    update_evolved()
    update_regression()

def get_history(symbol):
    i = 0
    l = [0]*16
    for record in db.execute("SELECT amount FROM modelHoldings WHERE symbol = ?", symbol):
        i+=record['amount']
        l.append(i)
        l.pop(0)
    return l



def make_evolved_predict(day):
    symbols = set([i['symbol'] for i in db.execute("SELECT DISTINCT symbol from ACTIONS")])
    amount = 10
    for symbol in symbols:
        action = network_utils.get_action(symbol, day, get_amount_owned(symbol),get_balance(symbol))
        # print(symbol)
        print(action)
        if action == 1:
            db.execute("INSERT INTO holdings (symbol, amount) VALUES (?, ?)",
                    symbol,
                    amount)
        elif action == 2:
            db.execute("INSERT INTO holdings (symbol, amount) VALUES (?, ?)",
                    symbol,
                    -get_amount_owned(symbol))
            
        db.execute("INSERT INTO ACTIONS (price, symbol, timestampe, action, amount) VALUES (?, ?, ?, ?, ?)",
                    network_utils.get_current_price(symbol, day),
                    symbol,
                    db.execute("""SELECT DATE('now', ?) as dates""", f'-{day} days')[0]["dates"],
                    action,
                    amount)
def make_regressive_predict(day):
    symbols = set([i['symbol'] for i in db.execute("SELECT DISTINCT symbol from modelACTIONS")])
    amount = 10
    # symbols = ["GOOGL"]
    for symbol in symbols:
        holdings = get_amount_owned_regress(symbol)
        action = model_utils.get_action(symbol, day, holdings, get_balance(symbol, True),get_history(symbol))
        if action == 1:
            db.execute("INSERT INTO modelHoldings (symbol, amount) VALUES (?, ?)",
                    symbol,
                    10)
        elif action == 2:
            db.execute("INSERT INTO modelHoldings (symbol, amount) VALUES (?, ?)",
                    symbol,
                    -10)
            
        db.execute("INSERT INTO modelACTIONS (price, symbol, timestampe, action, amount) VALUES (?, ?, ?, ?, ?)",
                    model_utils.get_current_price(symbol, day), # could use either ig
                    symbol,
                    db.execute("""SELECT DATE('now', ?) as dates""", f'-{day} days')[0]["dates"],
                    action,
                    10)


def make_predict(day):
    make_evolved_predict(day)
    make_regressive_predict(day)
        
    # stocks = db.execute("""SELECT * FROM stocks""")
    # for stock in stocks:
    #     value = lookup(stock["symbol"])
    #     if not value is None:
    #         db.execute(
    #             """UPDATE stocks SET price = ? WHERE id = ?""", value["price"], stock["id"]
#     #         )
# transactions(transactionid INTEGER PRIMARY KEY AUTOINCREMENT,
#                            price DECIMAL(10,2) NOT NULL,
#                            symbol VARCHAR(10) NOT NULL,
# timestamp NOT NULL DEFAULT(GETDATE()),
# # quantity REAL NOT NULL);CREATE TABLE ACTIONS(actionid INTEGER PRIMARY KEY AUTOINCREMENT, 
#    PRICE DECIMAL(10,2) NOT NULL,
#    symbol VARCHAR(10) NOT NULL,
#    timestampe NOT NULL DEFAULT CURRENT_TIMESTAMP,
#    action TINYINT,
#    amount INTEGER NOT NULL);
# scheduler = BlockingScheduler(timezone="Europe/Rome")
# scheduler.add_job(
#     func=update_stocks,
#     trigger="cron",
#     max_instances=1,
#     day_of_week='mon-sun',
#     hour=5,
#     minute=30
# )