import datetime
import math
from decimal import *
from dateutil.relativedelta import relativedelta
from dateutil.rrule import *


def repay():
    pass
    # 收到请求
    # 幂等 检查表是否存在数据 存在则返回
    # 计提检查
    # 还款试算
    # 保存请求及试算结果 防止跨日等操作的进行二次试算导致的差异
    # 还款--->同时用于轮询 保证接口唯一


def cal_repay_day(repay_day=None):
    """
    计算固定还款日
    :param repay_day:
    :return:
    """
    today = datetime.date.today()
    today = datetime.datetime.strptime('20210518', '%Y%m%d')
    if repay_day is None:  # 第一次贷款 1-28 29->25 30->26 31->27
        a, b = divmod(today.day, 29)
        if a == 0: repay_day = today.day
        if a == 1: repay_day = today.day - 28 + 24  # 29号以后从25日开始还款
        return f'{repay_day:0=2d}'
    else:
        return repay_day


def cal_repay_dates(repay_day, terms, first_min_days=0):
    """
    结算每期还款日
    :param repay_day: 固定还款日
    :param first_min_days: 最小还款间隔日
    :return:generator
    """
    today = datetime.date.today()
    today = datetime.datetime.strptime('20210518', '%Y%m%d')
    fist_term_date = datetime.datetime.strptime(
        (today + relativedelta(months=1)).strftime('%Y%m') + repay_day, '%Y%m%d'
    )
    if (fist_term_date - today).days < first_min_days:  # 小于首期最短日期 延后一期
        fist_term_date = fist_term_date + relativedelta(months=1)
    repay_dates = rrule(MONTHLY, count=terms, dtstart=fist_term_date)
    return repay_dates


def repay_plan_trail(loan_amt, terms, cal_way, day_rate, first_min_days=15, repay_day='02'):
    today = datetime.date.today()
    today = datetime.datetime.strptime('20210518', '%Y%m%d')
    repay_day = cal_repay_day(repay_day)
    repay_dates = cal_repay_dates(repay_day, terms, first_min_days)

    ratio = 1
    repay_plan = []
    last_cal_start_date = today
    for repay_date in repay_dates:
        interest_days = (repay_date - last_cal_start_date).days
        ratio = ratio / (1 + day_rate * interest_days)
        repay_plan.append([repay_date, ratio, interest_days])
        last_cal_start_date = repay_date
    term_amt = (loan_amt / sum([i[1] for i in repay_plan])).quantize(Decimal("0.00"), ROUND_DOWN)  # 向下取整
    remain_prin_amt = loan_amt
    for index, term in enumerate(repay_plan):
        int_amt = (remain_prin_amt * day_rate * term[2]).quantize(Decimal("0.00"), ROUND_HALF_UP)  # 四舍五入
        prin_amt = term_amt - int_amt
        repay_plan[index].extend([prin_amt, int_amt, term_amt])
        remain_prin_amt = remain_prin_amt - prin_amt  # 本金

    if remain_prin_amt > 0:
        repay_plan[-1][3] = repay_plan[-1][3] + remain_prin_amt
        repay_plan[-1][5] = repay_plan[-1][5] + remain_prin_amt

    return repay_plan


if __name__ == '__main__':
    loan_amt = 10000
    day_rate = Decimal('0.00020')
    terms = 36
    cal_way = '等额本息'
    repay_plan = repay_plan_trail(loan_amt=loan_amt, day_rate=day_rate, terms=terms, cal_way=cal_way)
