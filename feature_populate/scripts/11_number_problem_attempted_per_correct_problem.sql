-- Takes 10 seconds to execute (if index below is created!)
-- Created on July 1, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 11: number of problem attempted (feature 6) / number of correct problems (feature 8)


-- You need to create this index, otherwise it will take for ever
-- Takes 10 seconds to execute
-- ALTER TABLE `moocdb`.`user_long_feature`
-- ADD INDEX `user_week_idx` (`user_id` ASC, `dropout_feature_value_week` ASC) ;
set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)


SELECT 11,
	features.user_id,
	features.feature_week,
	CASE WHEN features.feature_value=0  then 0 else features2.feature_value  / features.feature_value end,
  @current_date
FROM `moocdb`.user_long_feature AS features,
	`moocdb`.user_long_feature AS features2
WHERE features.user_id = features2.user_id
	AND features.feature_week = features2.feature_week
	AND features.feature_id = 8
    #AND features.date_of_extraction >= @current_date
	AND features2.feature_id = 6
    #AND features2.date_of_extraction >= @current_date
;

