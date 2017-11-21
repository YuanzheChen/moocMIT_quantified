DROP TABLE if exists `moocdb`.`user_video_feature`;

CREATE TABLE `moocdb`.`user_video_feature` (
  `feature_value_id` INT NOT NULL AUTO_INCREMENT ,
  `feature_id` INT(3) NULL ,
  `user_id` VARCHAR(50) NULL ,
  `video_id` VARCHAR(255) NULL ,
  `feature_value` DOUBLE NULL ,
  `date_of_extraction` DATETIME NOT NULL ,
  PRIMARY KEY (`feature_value_id`),
  INDEX (`feature_id`, `user_id`, `video_id`)
);