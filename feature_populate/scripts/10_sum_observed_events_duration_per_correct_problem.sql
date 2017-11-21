-- Takes 12 seconds to execute (if index below is created!)
-- Created on July 1, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 10: total time spent on all resources during the week (feature 2) per Number of correct problems (feature 8)


-- You need to create this index, otherwise it will take for ever
-- Takes 10 seconds to execute
-- ALTER TABLE `moocdb`.`user_long_feature`
-- ADD INDEX `user_week_idx` (`user_id` ASC, `feature_week` ASC) ;

set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)


SELECT 10,
	user_long_feature.user_id,
	user_long_feature.feature_week,
	CASE WHEN user_long_feature.feature_value=0 then 0 else user_long_feature2.feature_value  / user_long_feature.feature_value end,
  @current_date
FROM `moocdb`.user_long_feature AS user_long_feature,
	`moocdb`.user_long_feature AS user_long_feature2
WHERE user_long_feature.user_id = user_long_feature2.user_id
	AND user_long_feature.feature_week = user_long_feature2.feature_week
	AND user_long_feature.feature_id = 8
    #AND user_long_feature.date_of_extraction >= @current_date
    AND user_long_feature2.feature_id = 2
    #AND user_long_feature2.date_of_extraction >= @current_date
;

