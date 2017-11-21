FROM quantified:base
RUN mkdir app
ADD . /app/MOOC-Learner-Quantified
WORKDIR /app/MOOC-Learner-Quantified
RUN ["chmod", "+x", "wait_for_it.sh"]
CMD ["./wait_for_it.sh", "curated", "python", "-u", "autorun.py", "-c", "../config/config.yml"]
