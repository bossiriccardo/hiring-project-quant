# hiring-project-quant

This is a coding exercise for the position of [Senior Quantitative Analyst at Reel](https://reelenergy-1727862339.teamtailor.com/jobs/6977095-senior-quantitative-analyst).

Please fork this repository and share your solved exercise to di@reel.energy and msl@reel.energy, Cc es@reel.energy.


### Background
Energinet, the Danish Transmission System Operator, is responsible for keeping the lights on in Denmark. That means maintaining the frequency in the power grid at about 50Hz and ensuring a constant balance between power supply and demand.
They achieve that by using several reserve markets: primary is very fast acting but moves small amounts of energy, tertiary is slower and shifts the most amount of energy, secondary is somewhere in between.
Secondary reserve is also known as aFRR (automatic frequency restoration reserve) and is the focus of this exercise.

Reserve activation patterns are very important in power trading as they influence the cost of imbalances, i.e. differences between the contracted positions and the physically produced/consumed energy in market parties' portfolios. High up-regulation reserve activations for example may lead to high imbalance prices.

### The task
Create a prediction model to forecast the aFRR activation on a minute-by-minute level for the next 8 settlement periods.

A settlement period or Market Time Unit (MTU) is a period of 15 minutes inside which imbalances are settled. 
We define `t0` as the current settlement period, i.e. the MTU starting at current time rounded down to nearest 15min and ending at current time rounded up to nearest 15min. For example, if now is 09:07 then `t0` will be the period between 09:00 and 09:15. 
Your model should be able to make predictions for `t1` to `t8` (included).

The model should output a timeseries of granularity 15 minutes and length of 8, where every element represents the forecasted aFRR activation on average in each MTU. 
The model should update its prediction at least every minute, based on live data published by Energinet from the source listed below.

Please also add a method that allows to backtest the performance.

You can see an example (very simple) implementation in `examples/persistence.py` where we use the average of the last 100 minutes of activations to predict the next 8 periods. This also shows you an example output format for the prediction model as well as the backtesting function.


### The data
You do not have to use other data sources in addition to the first one, but you are more than welcome to!

- Live minute-by-minute [power system view](https://www.energidataservice.dk/tso-electricity/PowerSystemRightNow) (including aFRR activations).
- Other [datasets](https://www.energidataservice.dk/organizations/tso-electricity).
