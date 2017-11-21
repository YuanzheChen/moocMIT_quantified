-- Takes 2 seconds to execute
-- Created on August 23, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 14: number of collaborations
set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;
set @start_date = 'START_DATE_PLACEHOLDER';


INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)

SELECT 14,
	user_dropout.user_id,
	FLOOR((UNIX_TIMESTAMP(collaborations.collaboration_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) AS week,
	COUNT(*) ,
    @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.collaborations AS collaborations
 ON collaborations.user_id = user_dropout.user_id
WHERE user_dropout.dropout_week IS NOT NULL
	AND FLOOR((UNIX_TIMESTAMP(collaborations.collaboration_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) < @num_weeks
GROUP BY user_dropout.user_id, week
HAVING week < @num_weeks
AND week >= 0
;

