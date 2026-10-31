FROM nginx:1.28-alpine
COPY static /usr/share/nginx/html
COPY docker/nginx.conf.template /etc/nginx/templates/default.conf.template
