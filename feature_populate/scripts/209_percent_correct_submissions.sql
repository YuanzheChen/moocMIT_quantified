-- Created on Feb 14, 2014
-- @author: Colin Taylor colin2328@gmail.com
-- Feature 209- Percentage of total submissions that were correct (feature 208 / feature 7)
-- Must have run populate_feature_208 and populate_feature_7 first!

set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)


SELECT 209,
	features.user_id,
	features.feature_week,
	CASE WHEN features.feature_value=0 then 0 else features2.feature_value  / features.feature_value end,
    @current_date
FROM `moocdb`.user_long_feature AS features,
	`moocdb`.user_long_feature AS features2
WHERE features.user_id = features2.user_id
	AND features.feature_week = features2.feature_week
	AND features.feature_id = 7
	AND features2.feature_id = 208
    AND features.date_of_extraction >= @current_date
    AND features2.date_of_extraction >= @current_date
;
