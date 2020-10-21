#pseudocode
#处理账户并发扣款问题
"""
流水-->流水表
try:
	exist = session.query(流水表)。filter(流水号).first()
	if exist:return exist.status
	else:
	    流水_data = data
	    session.add(流水_model(**data))
	    session.commit()
except:
	session.rollback()
	self.logger("交易失败")
	return "失败"
余额变动
try:
	rows = session.query(余额表)。filter(账号，变动后余额大于0).update({"amt":(余额_model.amt + amt)})
	if rows !=1:
        self.logger("交易失败")
        return "失败"
	session.query(流水表)。filter(流水号).update("status":"成功")
	commit
except:
	session.rollback()
	session.query(流水表)。filter(流水号).update("status":"失败")
	retuurn "失败"
return  "成功
"""
