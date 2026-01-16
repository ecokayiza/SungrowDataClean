# Temp Project for Timeseries


### 收益测算表 profit_metrics_info_df_cleaned.pkl
  - pvs_site_id：电站id
  - profit_increase_rate_percent：AI模式相比自发自用模式的收益提升率
  - pv_day_energy：日pv发电量（千瓦时）
  - load_day_energy：日负载发电量（千瓦时）
  - station_day_profit_ai_mode：AI模式日收益
  - station_day_profit_simulink_mode：自发自用模式日收益

### 电站属性表 filtered_station_info.pkl
  - pvs_site_id：电站id
  - lon，lat经纬度
  - inverter_pv_capacity + hybrid_inverter_pv_capacity：光伏装机功率
  - battery_capacity：储能电池容量

### station_data电站列表

- 每个文件夹是一个电站id = pvs_site_id = site_id
- ***_price.pkl：电价表**
  - price_type：1是馈网电价 2是购电电价  --> price_array
  - charge_type：4是动态电价 0是固定电价
- ***_point_data.pkl：遥测数据表**
  - 5分钟一个测点
  - load_forecast：负荷预测
  - pv_forecast:pv预测
  - pv_power：真实pv
  - load_power：真实负荷
  - battery_charge_power：电池当前充电量
  - battery_discharge_power：电池当前放电量	
  - battery_soc：电池当前soc	
  - available_battery_capacity：电池容量

## 任务：
将数据转为以天为单位的实例并用适合分析处理和深度学习的格式存储
ExpectedFormat(InstanceLevel):
Key: [station_id, date]
Property:[ battery_capacity,power_limit, charge_type?, profit_ai,profit_self]
Series:
  Daytime | Price_purchase | Price_sell | PV_forcast | PV_real | Load_forcast | Load_real | Battery_dis | Battery_cha | Battery_soc   