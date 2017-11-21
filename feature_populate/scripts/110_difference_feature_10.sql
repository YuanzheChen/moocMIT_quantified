-- Takes 7 seconds to execute
-- Created on July 2, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 110: difference of Feature 10 (total time spent on all resources during the week (feature 2) per Number of correct problems (feature 8))
-- 126339 rows

set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)


SELECT 110,
	features.user_id,
	features2.feature_week,
	-- features.feature_value,
	-- features2.feature_value,
  -- added this to fix divide by zero error
	IFNULL(features2.feature_value  / features.feature_value, 0),
    @current_date
FROM `moocdb`.user_long_feature AS features,
	`moocdb`.user_long_feature AS features2
WHERE
	-- same user
	features.user_id = features2.user_id
	-- 2 successive weeks
	AND features.feature_week = features2.feature_week - 1
	-- we are only interested in feature 5 (average length of forum posts)
	AND features.feature_id = 10
	AND features2.feature_id = 10
    #AND features.date_of_extraction >= @current_date
    #AND features2.date_of_extraction >= @current_date

-- LIMIT 1000
