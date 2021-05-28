import decimal
#四舍五入
"""
ROUND_CEILING 总是趋向无穷大向上取整
ROUND_DOWN　总是趋向0取整
ROUND_FLOOR 总是趋向负无穷大向下取整
ROUND_HALF_DOWN　如果最后一个有效数字大于或等于5则朝0反方向取整；否则，趋向0取整
ROUND_HALF_EVEN　类似于ROUND_HALF_DOWN，不过，如果最后一个有效数字值为5，则会检查前一位。偶数值会导致结果向下取整，奇数值导致结果向上取整
ROUND_HALF_UP 类似于ROUND_HALF_DOWN，不过如果最后一位有效数字为5，值会朝0的反方向取整
ROUND_UP　朝0的反方向取整
ROUND_05UP　如果最后一位是0或5，则朝0的反方向取整；否则向0取整
"""
context = decimal.getcontext()
context.rounding = decimal.ROUND_HALF_UP
print(round(decimal.Decimal(str(2.665)), 2))
print(round(decimal.Decimal(str(2.655)), 2))
