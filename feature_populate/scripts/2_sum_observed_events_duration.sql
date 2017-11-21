-- Takes 2900 seconds to execute
-- Created on June 30, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 2: total time spent on each resource during the week

set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;
set @start_date = 'START_DATE_PLACEHOLDER';

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)

SELECT 2,
	user_dropout.user_id,
	FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) AS week,
	SUM(observed_events.observed_event_duration),
    @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.observed_events AS observed_events
 ON observed_events.user_id = user_dropout.user_id
WHERE user_dropout.dropout_week IS NOT NULL
	-- AND user_dropout.user_id < 100
	AND FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) < @num_weeks
    AND observed_events.validity = 1
GROUP BY user_dropout.user_id, week
HAVING week < @num_weeks
AND week >= 0
;
