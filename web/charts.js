'use strict';

/*
 * Lightweight chart library for Model Monitoring.
 *
 * The application has no build step and no npm dependencies, so the charts are
 * rendered as SVG/HTML with the browser primitives.  Every chart reads its colours
 * from the shared design tokens (app.css) and re-renders on resize or theme change.
 * Line charts support opt-in monotone smoothing (`smooth`), gradient area fills
 * (`area`), automatic data markers and an optional generated legend (`legend`).
 */
(function (global) {
  var NS = 'http://www.w3.org/2000/svg';
  var charts = [];
  var resizeTimer = 0;
  var gradSeq = 0;
  var observer = null;

  function cssVar(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (value || '').trim() || fallback;
  }

  function palette() {
    return {
      blue: cssVar('--db-blue', '#2272b4'),
      green: cssVar('--db-green', '#0b7a55'),
      lava: cssVar('--db-lava', '#ff3621'),
      amber: cssVar('--db-amber', '#a85b00'),
      navy: cssVar('--db-navy', '#1b3139'),
      muted: cssVar('--db-muted', '#5f7078'),
      faint: cssVar('--db-faint', '#85959c'),
      line: cssVar('--db-line', '#dde2e5'),
      lineSoft: cssVar('--db-line-soft', '#eceff1'),
      surface: cssVar('--db-surface', '#ffffff'),
      surfaceAlt: cssVar('--db-surface-alt', '#fbfaf8'),
      ink: cssVar('--db-ink', '#11171c')
    };
  }

  function color(value, fallback) {
    var colors = palette();
    return colors[value] || value || fallback || colors.blue;
  }

  function svg(tag, attrs) {
    var node = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (attrs[key] !== null && attrs[key] !== undefined) {
        node.setAttribute(key, attrs[key]);
      }
    });
    return node;
  }

  function html(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function emptyState(container, message) {
    var box = html('div', 'lm-chart-empty');
    box.appendChild(html('span', 'lm-chart-empty-icon', '◔'));
    box.appendChild(html('span', '', message || 'Sin datos en el periodo seleccionado'));
    container.appendChild(box);
  }

  function fmtNumber(value, digits) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    return Number(value).toLocaleString('es-ES', {
      minimumFractionDigits: digits || 0,
      maximumFractionDigits: digits === undefined ? 2 : digits
    });
  }

  function fmtCompact(value) {
    var number = Number(value || 0);
    if (!isFinite(number)) return '—';
    var abs = Math.abs(number);
    if (abs >= 1000000) return (number / 1000000).toFixed(abs >= 10000000 ? 0 : 1) + 'M';
    if (abs >= 1000) return (number / 1000).toFixed(abs >= 10000 ? 0 : 1) + 'k';
    if (abs > 0 && abs < 1) return number.toFixed(2);
    return fmtNumber(number, abs % 1 ? 1 : 0);
  }

  function fmtMs(value) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    if (value >= 1000) return fmtNumber(value / 1000, 2) + ' s';
    return fmtNumber(value, value < 10 ? 1 : 0) + ' ms';
  }

  function fmtPercent(value, digits) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    return fmtNumber(value, digits === undefined ? 1 : digits) + '%';
  }

  /* Number('') and Number(null) both collapse to 0, so charts need an explicit
   * guard before trusting a value pulled out of an API payload. */
  function hasValue(value) {
    return value !== null && value !== undefined && value !== '' && isFinite(Number(value));
  }

  function dateValue(point) {
    if (point && point.t !== undefined && point.t !== null) return Number(point.t) * 1000;
    if (point && point.timestamp) return Number(point.timestamp) * 1000;
    return null;
  }

  function fmtDate(value, withTime) {
    var millis = typeof value === 'number' ? value : dateValue(value);
    if (millis === null || !isFinite(millis)) return '—';
    var date = new Date(millis);
    var options = withTime === false
      ? { day: '2-digit', month: 'short' }
      : { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' };
    return date.toLocaleString('es-ES', options);
  }

  function fmtAxisDate(value, spanSeconds) {
    var millis = typeof value === 'number' ? value : dateValue(value);
    if (millis === null || !isFinite(millis)) return '';
    var date = new Date(millis);
    if (spanSeconds <= 172800) {
      return date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString('es-ES', { day: '2-digit', month: 'short' });
  }

  function niceMax(value) {
    if (!value || value <= 0) return 1;
    var exponent = Math.floor(Math.log(value) / Math.LN10);
    var fraction = value / Math.pow(10, exponent);
    var nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;
    return nice * Math.pow(10, exponent);
  }

  function fmtCoord(value) {
    return isFinite(value) ? Number(value).toFixed(1) : '0';
  }

  /* The charts grow with the container unless the caller pins an explicit
   * height.  The aspect ratio stays bounded so a wide dashboard card does not
   * turn a sparkline into a wall of empty space; callers can tighten the bounds
   * with `minHeight` / `maxHeight` when a plot needs a different shape. */
  function chartHeight(width, spec) {
    spec = spec || {};
    if (spec.height) return spec.height;
    var ratio = spec.aspectRatio || 0.52;
    var min = spec.minHeight || 172;
    var max = spec.maxHeight || 352;
    if (max < min) max = min;
    return Math.max(min, Math.min(max, Math.round(width * ratio)));
  }

  /* The right gutter grows when end labels are requested and the bottom gutter
   * grows when the chart gets an explicit X axis caption.  End labels are
   * dropped on narrow cards: a 62px gutter would eat the plot on a phone. */
  function chartGeometry(width, height, spec) {
    spec = spec || {};
    var endLabels = spec.endLabels === true && width >= 460;
    var right = endLabels ? 62 : 14;
    var bottom = spec.xAxisTitle ? 60 : 52;
    return {
      left: 48,
      top: 16,
      width: width - 48 - right,
      height: height - bottom,
      endLabels: endLabels
    };
  }

  function linearPathD(segment) {
    return segment.map(function (point, index) {
      return (index === 0 ? 'M' : 'L') + fmtCoord(point.x) + ' ' + fmtCoord(point.y);
    }).join(' ');
  }

  /* Fritsch–Carlson monotone cubic interpolation.  The tangents are limited so
   * the curve never overshoots the sampled data, which keeps loss/metric curves
   * honest while still looking smoother than a polyline. */
  function smoothPathD(segment) {
    var count = segment.length;
    if (count < 3) return linearPathD(segment);
    var slopes = [];
    var tangents = [];
    var index;
    for (index = 0; index < count - 1; index += 1) {
      var dx = segment[index + 1].x - segment[index].x;
      var dy = segment[index + 1].y - segment[index].y;
      slopes.push(dx === 0 ? 0 : dy / dx);
    }
    tangents[0] = slopes[0];
    tangents[count - 1] = slopes[count - 2];
    for (index = 1; index < count - 1; index += 1) {
      var previous = slopes[index - 1];
      var next = slopes[index];
      if (previous * next <= 0) {
        tangents[index] = 0;
      } else {
        var average = (previous + next) / 2;
        tangents[index] = Math.abs(average) < Math.abs(previous) && Math.abs(average) < Math.abs(next)
          ? average
          : (Math.abs(previous) < Math.abs(next) ? previous : next);
      }
    }
    for (index = 0; index < count - 1; index += 1) {
      var delta = slopes[index];
      if (delta === 0) {
        tangents[index] = 0;
        tangents[index + 1] = 0;
        continue;
      }
      var alpha = tangents[index] / delta;
      var beta = tangents[index + 1] / delta;
      var magnitude = alpha * alpha + beta * beta;
      if (magnitude > 9) {
        var scale = 3 / Math.sqrt(magnitude);
        tangents[index] = scale * alpha * delta;
        tangents[index + 1] = scale * beta * delta;
      }
    }
    var path = 'M' + fmtCoord(segment[0].x) + ' ' + fmtCoord(segment[0].y);
    for (index = 0; index < count - 1; index += 1) {
      var start = segment[index];
      var end = segment[index + 1];
      var third = (end.x - start.x) / 3;
      path += ' C' + fmtCoord(start.x + third) + ' ' + fmtCoord(start.y + tangents[index] * third) +
        ' ' + fmtCoord(end.x - third) + ' ' + fmtCoord(end.y - tangents[index + 1] * third) +
        ' ' + fmtCoord(end.x) + ' ' + fmtCoord(end.y);
    }
    return path;
  }

  /* A vertical fade under the line reads as depth without adding visual noise. */
  function areaGradient(defs, stroke, opacity) {
    var id = 'lm-area-' + (gradSeq += 1);
    var gradient = svg('linearGradient', { id: id, x1: '0', y1: '0', x2: '0', y2: '1' });
    gradient.appendChild(svg('stop', {
      offset: '0%',
      'stop-color': stroke,
      'stop-opacity': opacity
    }));
    gradient.appendChild(svg('stop', {
      offset: '100%',
      'stop-color': stroke,
      'stop-opacity': 0
    }));
    defs.appendChild(gradient);
    return id;
  }

  function seriesLegend(container, series, spec) {
    if (spec.legend !== true || !series.length) return;
    var legend = html('div', 'lm-chart-legend');
    series.forEach(function (entry) {
      var item = html('span', 'lm-legend-item');
      var dot = html('span', 'lm-legend-dot');
      dot.style.background = color(entry.color, 'blue');
      if (entry.dashed) dot.classList.add('is-dashed');
      item.appendChild(dot);
      item.appendChild(document.createTextNode(entry.label || entry.key));
      legend.appendChild(item);
    });
    container.appendChild(legend);
  }

  function shell(container, spec) {
    clear(container);
    container.classList.add('lm-chart');
    var root = html('div', 'lm-chart-root');
    var tooltip = html('div', 'lm-chart-tooltip');
    tooltip.hidden = true;
    root.appendChild(tooltip);
    container.appendChild(root);
    return { root: root, tooltip: tooltip, spec: spec };
  }

  function showTooltip(context, event, title, rows) {
    var root = context.root;
    var tooltip = context.tooltip;
    clear(tooltip);
    tooltip.appendChild(html('div', 'lm-chart-tooltip-title', title));
    (rows || []).forEach(function (row) {
      var line = html('div', 'lm-chart-tooltip-row');
      var dot = html('span', 'lm-chart-tooltip-dot');
      dot.style.background = row.color || 'transparent';
      line.appendChild(dot);
      line.appendChild(html('span', 'lm-chart-tooltip-label', row.label));
      line.appendChild(html('span', 'lm-chart-tooltip-value', row.value));
      tooltip.appendChild(line);
    });
    tooltip.hidden = false;
    var bounds = root.getBoundingClientRect();
    var tipWidth = tooltip.offsetWidth || 0;
    var tipHeight = tooltip.offsetHeight || 0;
    var x = event.clientX - bounds.left + 12;
    var y = event.clientY - bounds.top + 12;
    /* Flip to the other side of the cursor instead of letting the card clip the
     * tooltip at the edge of a narrow plot. */
    if (x + tipWidth > bounds.width - 4) x = event.clientX - bounds.left - tipWidth - 12;
    if (y + tipHeight > bounds.height - 4) y = event.clientY - bounds.top - tipHeight - 12;
    tooltip.style.left = Math.max(4, Math.min(x, bounds.width - tipWidth - 4)) + 'px';
    tooltip.style.top = Math.max(4, Math.min(y, bounds.height - tipHeight - 4)) + 'px';
  }

  function hideTooltip(context) {
    context.tooltip.hidden = true;
  }

  function axes(group, spec, geometry) {
    var colors = palette();
    var ticks = 4;
    var minValue = spec.minValue || 0;
    var maxValue = spec.maxValue || 1;
    var range = spec.range || (maxValue - minValue) || 1;
    for (var index = 0; index <= ticks; index += 1) {
      var y = geometry.top + geometry.height - (geometry.height * index / ticks);
      group.appendChild(
        svg('line', {
          x1: geometry.left,
          x2: geometry.left + geometry.width,
          y1: y,
          y2: y,
          stroke: index === 0 ? colors.line : colors.lineSoft,
          'stroke-width': 1
        })
      );
      var value = minValue + range * index / ticks;
      var label = svg('text', {
        x: geometry.left - 8,
        y: y + 4,
        'class': 'lm-axis-tick',
        'text-anchor': 'end',
        fill: colors.faint,
        'font-size': 10.5
      });
      label.textContent = (spec.formatAxis || fmtCompact)(value);
      group.appendChild(label);
    }
    /* Tick density follows the available width so labels never collide. */
    var count = Math.min(
      spec.maxXTicks || 6,
      Math.max(2, Math.floor(geometry.width / 78)),
      geometry.points.length
    );
    for (var tick = 0; tick < count; tick += 1) {
      var indexPoint = count === 1
        ? 0
        : Math.round((geometry.points.length - 1) * tick / (count - 1));
      var point = geometry.points[indexPoint];
      if (!point) continue;
      var x = geometry.left + (geometry.width * indexPoint / Math.max(1, geometry.points.length - 1));
      var text = svg('text', {
        x: x,
        y: geometry.top + geometry.height + 17,
        'class': 'lm-axis-tick lm-axis-tick-x',
        'text-anchor': tick === count - 1 ? 'end' : tick === 0 ? 'start' : 'middle',
        fill: colors.faint,
        'font-size': 10.5
      });
      text.textContent = spec.formatXAxis ? spec.formatXAxis(point) : fmtAxisDate(point.t, geometry.span);
      group.appendChild(text);
    }
    if (spec.xAxisTitle) {
      var caption = svg('text', {
        x: geometry.left + geometry.width / 2,
        y: geometry.top + geometry.height + 36,
        'class': 'lm-axis-title',
        'text-anchor': 'middle',
        fill: colors.faint,
        'font-size': 9.5,
        'letter-spacing': '0.09em'
      });
      caption.textContent = String(spec.xAxisTitle).toUpperCase();
      group.appendChild(caption);
    }
  }

  function lineChart(container, spec) {
    spec = spec || {};
    var points = spec.points || [];
    var series = spec.series || [];
    var hasData = points.some(function (point) {
      return series.some(function (entry) {
        return hasValue(point[entry.key]);
      });
    });
    if (!points.length || !hasData) {
      container.classList.add('lm-chart');
      clear(container);
      emptyState(container, spec.emptyText);
      container._lmRender = null;
      return;
    }
    var context = shell(container, spec);
    var width = Math.max(320, container.clientWidth || 640);
    var height = chartHeight(width, spec);
    var geometry = chartGeometry(width, height, spec);
    var node = svg('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      width: '100%',
      height: height,
      role: 'img',
      'aria-label': spec.ariaLabel || spec.title || 'Gráfico de series temporales'
    });
    var title = svg('title', {});
    title.textContent = spec.title || 'Serie temporal';
    node.appendChild(title);
    var defs = svg('defs', {});
    node.appendChild(defs);
    var smooth = spec.smooth === true;
    var colors = palette();
    geometry.points = points;
    geometry.span = points.length > 1
      ? Number(points[points.length - 1].t) - Number(points[0].t)
      : 60;
    var maxValue = 0;
    var minValue = 0;
    points.forEach(function (point) {
      series.forEach(function (entry) {
        if (!hasValue(point[entry.key])) return;
        var value = Number(point[entry.key]);
        if (value > maxValue) maxValue = value;
        if (value < minValue) minValue = value;
      });
    });
    var niceMaxValue = niceMax(maxValue * 1.08 || 1);
    var niceMinValue = minValue < 0 ? -niceMax(-minValue * 1.08) : 0;
    var scaleSpec = {
      minValue: niceMinValue,
      maxValue: niceMaxValue,
      range: (niceMaxValue - niceMinValue) || 1,
      formatAxis: spec.formatAxis || fmtCompact,
      formatXAxis: spec.formatXAxis,
      xAxisTitle: spec.xAxisTitle,
      maxXTicks: spec.maxXTicks
    };
    var background = svg('g', {});
    axes(background, scaleSpec, geometry);
    node.appendChild(background);

    function xAt(index) {
      return geometry.left + (geometry.width * index / Math.max(1, points.length - 1));
    }

    function yAt(value) {
      return geometry.top + geometry.height -
        (geometry.height * (Number(value) - scaleSpec.minValue) / scaleSpec.range);
    }

    var order = points.map(function (point, index) {
      return { point: point, index: index };
    });
    var markerGroup = svg('g', { 'class': 'lm-chart-markers' });
    var markerList = [];
    var endLabelList = [];
    series.forEach(function (entry) {
      var stroke = color(entry.color, 'blue');
      var segments = [];
      var current = [];
      order.forEach(function (item) {
        var value = item.point[entry.key];
        if (!hasValue(value)) {
          if (current.length > 1) segments.push(current);
          current = [];
          return;
        }
        current.push({ x: xAt(item.index), y: yAt(value), value: Number(value) });
      });
      if (current.length) segments.push(current);
      var gradientId = entry.area
        ? areaGradient(defs, stroke, entry.areaOpacity === undefined ? 0.18 : entry.areaOpacity)
        : null;
      segments.forEach(function (segment) {
        var path = smooth ? smoothPathD(segment) : linearPathD(segment);
        if (entry.area) {
          var baseline = geometry.top + geometry.height;
          var area = path +
            ' L' + fmtCoord(segment[segment.length - 1].x) + ' ' + fmtCoord(baseline) +
            ' L' + fmtCoord(segment[0].x) + ' ' + fmtCoord(baseline) + ' Z';
          node.appendChild(
            svg('path', {
              d: area,
              fill: 'url(#' + gradientId + ')',
              stroke: 'none',
              'class': 'lm-area'
            })
          );
        }
        node.appendChild(
          svg('path', {
            d: path,
            'class': 'lm-line',
            fill: 'none',
            stroke: stroke,
            'stroke-width': entry.width || 2,
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round',
            'stroke-dasharray': entry.dashed ? '5 4' : null
          })
        );
        /* Dashed percentile lines stay uncluttered: only the head is marked. */
        if (segment.length <= 16 && !entry.dashed) {
          segment.forEach(function (point) {
            markerList.push({ point: point, stroke: stroke, small: true });
          });
        } else {
          markerList.push({ point: segment[segment.length - 1], stroke: stroke, small: false });
        }
      });
      if (geometry.endLabels) {
        for (var tailIndex = segments.length - 1; tailIndex >= 0; tailIndex -= 1) {
          var tail = segments[tailIndex];
          if (tail.length) {
            endLabelList.push({
              point: tail[tail.length - 1],
              stroke: stroke,
              format: entry.formatValue
            });
            break;
          }
        }
      }
    });
    markerList.forEach(function (marker) {
      var attrs = {
        cx: fmtCoord(marker.point.x),
        cy: fmtCoord(marker.point.y),
        r: marker.small ? 2.3 : 3.2,
        fill: marker.small ? colors.surface : marker.stroke,
        stroke: marker.small ? marker.stroke : colors.surface,
        'stroke-width': 1.5
      };
      if (!marker.small) attrs['class'] = 'lm-marker-end';
      markerGroup.appendChild(svg('circle', attrs));
    });
    node.appendChild(markerGroup);
    endLabelList.forEach(function (entry) {
      var label = svg('text', {
        x: fmtCoord(entry.point.x + 7),
        y: fmtCoord(entry.point.y + 3.5),
        'class': 'lm-end-label',
        'text-anchor': 'start',
        fill: entry.stroke,
        stroke: colors.surface,
        'stroke-width': 3,
        'paint-order': 'stroke',
        'stroke-linejoin': 'round',
        'font-size': 10,
        'font-weight': 700
      });
      label.textContent = (entry.format || spec.formatValue || fmtNumber)(entry.point.value);
      node.appendChild(label);
    });


    var guide = svg('line', {
      y1: geometry.top,
      y2: geometry.top + geometry.height,
      stroke: colors.faint,
      'stroke-width': 1,
      'stroke-dasharray': '3 3',
      visibility: 'hidden'
    });
    node.appendChild(guide);
    var highlight = svg('g', {});
    node.appendChild(highlight);
    var overlay = svg('rect', {
      x: geometry.left,
      y: geometry.top,
      width: geometry.width,
      height: geometry.height,
      fill: 'transparent',
      style: 'cursor:crosshair'
    });
    node.appendChild(overlay);

    function indexAt(event) {
      var bounds = node.getBoundingClientRect();
      var relative = (event.clientX - bounds.left) / bounds.width * width;
      return Math.max(
        0,
        Math.min(
          points.length - 1,
          Math.round((relative - geometry.left) / geometry.width * (points.length - 1))
        )
      );
    }

    function inspect(event) {
      var index = indexAt(event);
      var point = points[index];
      var x = xAt(index);
      guide.setAttribute('x1', x);
      guide.setAttribute('x2', x);
      guide.setAttribute('visibility', 'visible');
      clear(highlight);
      series.forEach(function (entry) {
        var value = point[entry.key];
        if (!hasValue(value)) return;
        highlight.appendChild(
          svg('circle', {
            cx: x,
            cy: yAt(value),
            r: 3.5,
            fill: color(entry.color, 'blue'),
            stroke: colors.surface,
            'stroke-width': 1.5
          })
        );
      });
      showTooltip(
        context,
        event,
        (spec.tooltipTitle || fmtDate)(point),
        series.map(function (entry) {
          return {
            color: color(entry.color, 'blue'),
            label: entry.label,
            /* A series may carry its own formatter (fractions shown as
             * percentages, counts as integers, ...). */
            value: (entry.formatValue || spec.formatValue || fmtNumber)(point[entry.key])
          };
        })
      );
    }

    overlay.addEventListener('mousemove', inspect);
    overlay.addEventListener('mouseleave', function () {
      hideTooltip(context);
      guide.setAttribute('visibility', 'hidden');
      clear(highlight);
    });
    overlay.addEventListener('click', function (event) {
      if (spec.onSelect) spec.onSelect(points[indexAt(event)]);
    });
    context.root.insertBefore(node, context.tooltip);
    seriesLegend(container, series, spec);
    container._lmRender = function () { lineChart(container, spec); };
  }


  function stackedBars(container, spec) {
    spec = spec || {};
    var points = spec.points || [];
    var series = spec.series || [];
    var hasData = points.some(function (point) {
      return series.some(function (entry) {
        var value = Number(point[entry.key]);
        return isFinite(value) && value > 0;
      });
    });
    if (!points.length || !hasData) {
      container.classList.add('lm-chart');
      clear(container);
      emptyState(container, spec.emptyText);
      container._lmRender = null;
      return;
    }
    var context = shell(container, spec);
    var width = Math.max(320, container.clientWidth || 640);
    var height = chartHeight(width, spec);
    var geometry = chartGeometry(width, height, spec);
    geometry.points = points;
    geometry.span = points.length > 1
      ? Number(points[points.length - 1].t) - Number(points[0].t)
      : 60;
    var maxValue = 0;
    points.forEach(function (point) {
      var total = series.reduce(function (sum, entry) {
        var value = Number(point[entry.key]);
        return sum + (isFinite(value) && value > 0 ? value : 0);
      }, 0);
      if (total > maxValue) maxValue = total;
    });
    var scaleSpec = {
      maxValue: niceMax(maxValue * 1.08 || 1),
      formatAxis: spec.formatAxis || fmtCompact,
      xAxisTitle: spec.xAxisTitle,
      maxXTicks: spec.maxXTicks
    };
    var node = svg('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      width: '100%',
      height: height,
      role: 'img',
      'aria-label': spec.ariaLabel || 'Solicitudes por intervalo'
    });
    var title = svg('title', {});
    title.textContent = spec.title || 'Solicitudes';
    node.appendChild(title);
    var background = svg('g', {});
    axes(background, scaleSpec, geometry);
    node.appendChild(background);

    var slot = geometry.width / points.length;
    var barWidth = Math.max(1, Math.min(spec.maxBarWidth || 22, slot * 0.68));
    var bars = svg('g', {});
    points.forEach(function (point, index) {
      var x = geometry.left + slot * index + (slot - barWidth) / 2;
      var y = geometry.top + geometry.height;
      series.forEach(function (entry) {
        var value = Number(point[entry.key]);
        if (!isFinite(value) || value <= 0) return;
        var barHeight = geometry.height * value / scaleSpec.maxValue;
        y -= barHeight;
        bars.appendChild(
          svg('rect', {
            x: x,
            y: y,
            width: barWidth,
            height: Math.max(1, barHeight),
            fill: color(entry.color, 'blue'),
            rx: Math.min(2, barWidth / 3)
          })
        );
      });
    });
    node.appendChild(bars);

    var guide = svg('line', {
      y1: geometry.top,
      y2: geometry.top + geometry.height,
      stroke: palette().faint,
      'stroke-width': 1,
      'stroke-dasharray': '3 3',
      visibility: 'hidden'
    });
    node.appendChild(guide);
    var overlay = svg('rect', {
      x: geometry.left,
      y: geometry.top,
      width: geometry.width,
      height: geometry.height,
      fill: 'transparent',
      style: 'cursor:crosshair'
    });
    node.appendChild(overlay);

    function indexAt(event) {
      var bounds = node.getBoundingClientRect();
      var relative = (event.clientX - bounds.left) / bounds.width * width;
      return Math.max(
        0,
        Math.min(points.length - 1, Math.floor((relative - geometry.left) / slot))
      );
    }

    function inspect(event) {
      var index = indexAt(event);
      var point = points[index];
      var x = geometry.left + slot * index + slot / 2;
      guide.setAttribute('x1', x);
      guide.setAttribute('x2', x);
      guide.setAttribute('visibility', 'visible');
      var rows = series.map(function (entry) {
        return {
          color: color(entry.color, 'blue'),
          label: entry.label,
          value: (spec.formatValue || fmtNumber)(point[entry.key])
        };
      });
      if (series.length > 1) {
        var total = series.reduce(function (sum, entry) {
          var value = Number(point[entry.key]);
          return sum + (isFinite(value) ? value : 0);
        }, 0);
        rows.push({
          color: 'transparent',
          label: 'Total',
          value: (spec.formatValue || fmtNumber)(total)
        });
      }
      showTooltip(context, event, (spec.tooltipTitle || fmtDate)(point), rows);
    }

    overlay.addEventListener('mousemove', inspect);
    overlay.addEventListener('mouseleave', function () {
      hideTooltip(context);
      guide.setAttribute('visibility', 'hidden');
    });
    overlay.addEventListener('click', function (event) {
      if (spec.onSelect) spec.onSelect(points[indexAt(event)]);
    });
    context.root.insertBefore(node, context.tooltip);
    seriesLegend(container, series, spec);
    container._lmRender = function () { stackedBars(container, spec); };
  }


  function hbar(container, spec) {
    spec = spec || {};
    var items = (spec.items || []).slice(0, spec.limit || 8);
    clear(container);
    container.classList.add('lm-chart');
    if (!items.length) {
      emptyState(container, spec.emptyText);
      container._lmRender = null;
      return;
    }
    var max = items.reduce(function (top, item) {
      var value = Number(item.value);
      return isFinite(value) ? Math.max(top, value) : top;
    }, 0) || 1;
    var wrap = html('div', 'lm-hbar');
    items.forEach(function (item) {
      var row = html('button', 'lm-hbar-row');
      row.type = 'button';
      if (spec.onSelect) {
        row.addEventListener('click', function () { spec.onSelect(item); });
      } else {
        row.disabled = true;
      }
      var head = html('div', 'lm-hbar-head');
      head.appendChild(html('span', 'lm-hbar-label', item.label));
      head.appendChild(
        html('span', 'lm-hbar-value', (spec.formatValue || fmtNumber)(item.value))
      );
      row.appendChild(head);
      var track = html('div', 'lm-hbar-track');
      var fill = html('div', 'lm-hbar-fill');
      var numeric = Number(item.value);
      var ratio = isFinite(numeric) ? numeric / max : 0;
      fill.style.width = Math.max(1.5, ratio * 100) + '%';
      if (item.color) fill.style.background = color(item.color);
      track.appendChild(fill);
      row.appendChild(track);
      if (item.meta) row.appendChild(html('div', 'lm-hbar-meta', item.meta));
      wrap.appendChild(row);
    });
    container.appendChild(wrap);
    if (spec.legend && spec.legend.length) {
      var legend = html('div', 'lm-chart-legend');
      spec.legend.forEach(function (entry) {
        var item = html('span', 'lm-legend-item');
        var dot = html('span', 'lm-legend-dot');
        dot.style.background = color(entry.color);
        item.appendChild(dot);
        item.appendChild(document.createTextNode(entry.label));
        legend.appendChild(item);
      });
      container.appendChild(legend);
    }
    container._lmRender = function () { hbar(container, spec); };
  }

  function donut(container, spec) {
    spec = spec || {};
    var items = (spec.items || []).filter(function (item) {
      return hasValue(item.value) && Number(item.value) > 0;
    });
    clear(container);
    container.classList.add('lm-chart');
    if (!items.length) {
      emptyState(container, spec.emptyText);
      container._lmRender = null;
      return;
    }
    var total = items.reduce(function (sum, item) { return sum + Number(item.value); }, 0);
    var size = spec.size || 168;
    var radius = size / 2 - 12;
    var circumference = 2 * Math.PI * radius;
    var node = svg('svg', {
      viewBox: '0 0 ' + size + ' ' + size,
      width: size,
      height: size,
      role: 'img',
      'aria-label': spec.ariaLabel || 'Distribución'
    });
    var context = shell(container, spec);
    var group = svg('g', {
      transform: 'translate(' + size / 2 + ' ' + size / 2 + ') rotate(-90)'
    });
    var offset = 0;
    items.forEach(function (item) {
      var fraction = Number(item.value) / total;
      var circle = svg('circle', {
        r: radius,
        fill: 'none',
        stroke: color(item.color, 'blue'),
        'stroke-width': spec.thickness || 16,
        'stroke-dasharray': (fraction * circumference) + ' ' + circumference,
        'stroke-dashoffset': -offset
      });
      circle.style.cursor = spec.onSelect ? 'pointer' : 'default';
      circle.addEventListener('mousemove', function (event) {
        showTooltip(context, event, item.label, [
          {
            color: color(item.color, 'blue'),
            label: (spec.formatValue || fmtNumber)(item.value),
            value: fmtPercent(fraction * 100)
          }
        ]);
      });
      circle.addEventListener('mouseleave', function () { hideTooltip(context); });
      if (spec.onSelect) {
        circle.addEventListener('click', function () { spec.onSelect(item); });
      }
      group.appendChild(circle);
      offset += fraction * circumference;
    });
    node.appendChild(group);
    var center = svg('text', {
      x: size / 2,
      y: size / 2 - 2,
      'text-anchor': 'middle',
      fill: palette().ink,
      'font-size': 19,
      'font-weight': 600
    });
    center.textContent = (spec.formatCenter || fmtCompact)(total);
    node.appendChild(center);
    var caption = svg('text', {
      x: size / 2,
      y: size / 2 + 16,
      'text-anchor': 'middle',
      fill: palette().faint,
      'font-size': 10
    });
    caption.textContent = spec.centerLabel || 'total';
    node.appendChild(caption);
    context.root.insertBefore(node, context.tooltip);
    var legend = html('div', 'lm-donut-legend');
    items.forEach(function (item) {
      var entry = html('div', 'lm-donut-legend-item');
      var dot = html('span', 'lm-legend-dot');
      dot.style.background = color(item.color, 'blue');
      entry.appendChild(dot);
      entry.appendChild(html('span', 'lm-donut-label', item.label));
      entry.appendChild(
        html('span', 'lm-donut-value', (spec.formatValue || fmtNumber)(item.value))
      );
      legend.appendChild(entry);
    });
    container.appendChild(legend);
    container._lmRender = function () { donut(container, spec); };
  }


  function isLive(container) {
    return typeof container.isConnected === 'boolean'
      ? container.isConnected
      : document.documentElement.contains(container);
  }

  function redrawAll() {
    /* Views are re-rendered by replacing their markup, so the registry has to
     * drop detached containers instead of holding on to stale closures. */
    charts = charts.filter(function (container) {
      if (isLive(container)) return true;
      if (observer && container._lmObserved) {
        observer.unobserve(container);
        container._lmObserved = false;
      }
      return false;
    });
    charts.forEach(function (container) {
      if (container._lmRender) container._lmRender();
    });
  }

  /* The sidebar collapses and the dashboard cards reflow without a window
   * resize event, so every chart watches its own box when the browser supports
   * it.  Only width changes trigger a redraw: height changes are a consequence
   * of the redraw and would otherwise loop. */
  function watch(container) {
    if (typeof window.ResizeObserver !== 'function') return;
    if (!observer) {
      observer = new window.ResizeObserver(function (entries) {
        var changed = false;
        entries.forEach(function (entry) {
          var width = Math.round(entry.contentRect.width);
          if (!width || width === entry.target._lmWidth) return;
          entry.target._lmWidth = width;
          changed = true;
        });
        if (changed) {
          window.clearTimeout(resizeTimer);
          resizeTimer = window.setTimeout(redrawAll, 80);
        }
      });
    }
    if (container._lmObserved) return;
    container._lmObserved = true;
    container._lmWidth = Math.round(container.clientWidth || 0);
    observer.observe(container);
  }

  function register(container) {
    if (!container) return;
    container.setAttribute('data-lm-chart', '1');
    if (charts.indexOf(container) === -1) charts.push(container);
    watch(container);
  }

  /* Fallback for browsers without ResizeObserver. */
  if (typeof window.ResizeObserver !== 'function') {
    window.addEventListener('resize', function () {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(redrawAll, 160);
    });
  }

  /* Keep the observer's bookkeeping in sync with app-driven renders so the
   * initial observation of an unchanged box does not trigger a second paint. */
  function sync(container) {
    container._lmWidth = Math.round(container.clientWidth || 0);
  }

  global.LMCharts = {
    line: function (container, spec) { register(container); lineChart(container, spec); sync(container); },
    stackedBars: function (container, spec) { register(container); stackedBars(container, spec); sync(container); },
    bars: function (container, spec) { register(container); hbar(container, spec); sync(container); },
    donut: function (container, spec) { register(container); donut(container, spec); sync(container); },
    redrawAll: redrawAll,
    palette: palette,
    fmtNumber: fmtNumber,
    fmtCompact: fmtCompact,
    fmtMs: fmtMs,
    fmtPercent: fmtPercent,
    fmtDate: fmtDate,
    fmtAxisDate: fmtAxisDate
  };
})(window);


