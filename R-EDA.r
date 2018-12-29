
library('ggplot2')
library('car')
library('corrplot')

library('psych')

data2 = read.csv('/Users/asayou/Desktop/DSBA/tap4fun竞赛数据/dataR2.csv')

head(data2)

purchase = data2$prediction_pay_price
describe(purchase)

ggplot(data2) + geom_boxplot(aes(y=prediction_pay_price)) + ylim(0,100) + ylab('') + xlab('purchase')


pairs(~ basic_material_add + acceleration_add + proUnit_add +
      building_level + research_level + active_pvp_winrate +
      pvp_lanch_count + prediction_pay_price, data=data2,
      pch=c(1,2,3,4,5)[data2$pay_class])      

data3 = data2[,c(-1,-12,-13,-14,-15)]

corrplot(cor(data3))
