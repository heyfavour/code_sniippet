import decimal
#四舍五入
context = decimal.getcontext()
context.rounding = decimal.ROUND_HALF_UP
print(round(decimal.Decimal(str(2.665)), 2))
print(round(decimal.Decimal(str(2.655)), 2))
