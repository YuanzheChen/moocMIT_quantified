-- Takes 300 seconds to execute
-- Created on August 23, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 17: Total time spent on all resources - book only - during the week

set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;
set @start_date = 'START_DATE_PLACEHOLDER';

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)


SELECT 17,
	user_dropout.user_id,
	FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) AS week,
	SUM(observed_events.observed_event_duration),
    @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.observed_events AS observed_events
 ON observed_events.user_id = user_dropout.user_id
INNER JOIN `moocdb`.urls AS urls
 ON urls.url_id=observed_events.url_id
INNER JOIN `moocdb`.resources AS resources
 ON resources.resource_uri = urls.url
INNER JOIN `moocdb`.resource_types AS resource_types
 ON resource_types.resource_type_id = resources.resource_type_id
WHERE user_dropout.dropout_week IS NOT NULL
	-- AND user_dropout.user_id < 100
	AND resource_types.resource_type_id = 3
	AND FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) < @num_weeks
    AND observed_events.validity = 1
GROUP BY user_dropout.user_id, week
HAVING week < @num_weeks
AND week >= 0
;

