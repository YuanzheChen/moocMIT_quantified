DROP TABLE if exists `moocdb`.`feature_info`;

CREATE TABLE `moocdb`.`feature_info` (
  `feature_id` INT(5) NOT NULL,
  `feature_table` VARCHAR(30) NOT NULL ,
  `feature_name` TINYTEXT NULL ,
  `feature_description` TEXT NULL ,
  PRIMARY KEY (`feature_id`)
);
