-- Takes 1 seconds to execute
-- Created on July 2, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 105: difference of Feature 5 (average length of forum posts)
-- 2433 rows
set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)

SELECT 105,
	features.user_id,
	features2.feature_week,
	-- features.feature_value,
	-- features2.feature_value,
  -- added this to fix divide by zero error
	IFNULL(features2.feature_value  / features.feature_value,0),
    @current_date
FROM `moocdb`.user_long_feature AS features,
	`moocdb`.user_long_feature AS features2,
	`moocdb`.user_long_feature AS features3

WHERE
	-- same user
	features.user_id = features2.user_id
	-- 2 successive weeks
	AND features.feature_week = features2.feature_week - 1
	-- we are only interested in feature 5 (average length of forum posts)
	AND features.feature_id = 5
	AND features2.feature_id = 5
    AND features.date_of_extraction >= @current_date
    AND features2.date_of_extraction >= @current_date
-- LIMIT 1000
