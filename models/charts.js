'use strict';

/*
 * Lightweight chart library for Model Monitoring.
 *
 * The application has no build step and no npm dependencies, so the charts are
 * rendered as SVG/HTML with the browser primitives.  Every chart reads its colours
 * from the shared design tokens (app.css) and re-renders on resize or theme change.
 */
(function (global) {
  var NS = 'http://www.w3.org/2000/svg';
  var charts = [];
  var resizeTimer = 0;

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
    var x = event.clientX - bounds.left + 12;
    var y = event.clientY - bounds.top + 12;
    tooltip.style.left = Math.max(4, Math.min(x, bounds.width - tooltip.offsetWidth - 4)) + 'px';
    tooltip.style.top = Math.max(4, Math.min(y, bounds.height - tooltip.offsetHeight - 4)) + 'px';
  }

  function hideTooltip(context) {
    context.tooltip.hidden = true;
  }

  function axes(group, spec, geometry) {
    var colors = palette();
    var ticks = 4;
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
      var value = (spec.maxValue || 1) * index / ticks;
      var label = svg('text', {
        x: geometry.left - 8,
        y: y + 4,
        'text-anchor': 'end',
        fill: colors.faint,
        'font-size': 10
      });
      label.textContent = (spec.formatAxis || fmtCompact)(value);
      group.appendChild(label);
    }
    var count = Math.min(6, geometry.points.length);
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
        'text-anchor': tick === count - 1 ? 'end' : tick === 0 ? 'start' : 'middle',
        fill: colors.faint,
        'font-size': 10
      });
      text.textContent = fmtAxisDate(point.t, geometry.span);
      group.appendChild(text);
    }
  }

  function lineChart(container, spec) {
    spec = spec || {};
    var points = spec.points || [];
    var series = spec.series || [];
    var hasData = points.some(function (point) {
      return series.some(function (entry) {
        return point[entry.key] !== null && point[entry.key] !== undefined;
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
    var height = spec.height || 250;
    var geometry = { left: 48, top: 16, width: width - 62, height: height - 52 };
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
    geometry.points = points;
    geometry.span = points.length > 1
      ? Number(points[points.length - 1].t) - Number(points[0].t)
      : 60;
    var maxValue = 0;
    points.forEach(function (point) {
      series.forEach(function (entry) {
        var value = Number(point[entry.key]);
        if (isFinite(value) && value > maxValue) maxValue = value;
      });
    });
    var scaleSpec = {
      maxValue: niceMax(maxValue * 1.08 || 1),
      formatAxis: spec.formatAxis || fmtCompact
    };
    var background = svg('g', {});
    axes(background, scaleSpec, geometry);
    node.appendChild(background);

    function xAt(index) {
      return geometry.left + (geometry.width * index / Math.max(1, points.length - 1));
    }

    function yAt(value) {
      return geometry.top + geometry.height -
        (geometry.height * Number(value) / scaleSpec.maxValue);
    }

    var order = points.map(function (point, index) {
      return { point: point, index: index };
    });
    series.forEach(function (entry) {
      var stroke = color(entry.color, 'blue');
      var segments = [];
      var current = [];
      order.forEach(function (item) {
        var value = item.point[entry.key];
        if (value === null || value === undefined || !isFinite(Number(value))) {
          if (current.length > 1) segments.push(current);
          current = [];
          return;
        }
        current.push({ x: xAt(item.index), y: yAt(value) });
      });
      if (current.length) segments.push(current);
      segments.forEach(function (segment) {
        var path = segment.map(function (point, index) {
          return (index === 0 ? 'M' : 'L') + point.x.toFixed(1) + ' ' + point.y.toFixed(1);
        }).join(' ');
        if (entry.area) {
          var baseline = geometry.top + geometry.height;
          var area = path +
            ' L' + segment[segment.length - 1].x.toFixed(1) + ' ' + baseline +
            ' L' + segment[0].x.toFixed(1) + ' ' + baseline + ' Z';
          node.appendChild(
            svg('path', {
              d: area,
              fill: stroke,
              opacity: entry.areaOpacity === undefined ? 0.12 : entry.areaOpacity,
              stroke: 'none'
            })
          );
        }
        node.appendChild(
          svg('path', {
            d: path,
            fill: 'none',
            stroke: stroke,
            'stroke-width': entry.width || 2,
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round',
            'stroke-dasharray': entry.dashed ? '5 4' : null
          })
        );
      });
    });


    var guide = svg('line', {
      y1: geometry.top,
      y2: geometry.top + geometry.height,
      stroke: palette().faint,
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
        if (value === null || value === undefined || !isFinite(Number(value))) return;
        highlight.appendChild(
          svg('circle', {
            cx: x,
            cy: yAt(value),
            r: 3.5,
            fill: color(entry.color, 'blue'),
            stroke: palette().surface,
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
            value: (spec.formatValue || fmtNumber)(point[entry.key])
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
    container._lmRender = function () { lineChart(container, spec); };
  }


  function stackedBars(container, spec) {
    spec = spec || {};
    var points = spec.points || [];
    var series = spec.series || [];
    var hasData = points.some(function (point) {
      return series.some(function (entry) { return Number(point[entry.key]) > 0; });
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
    var height = spec.height || 250;
    var geometry = { left: 48, top: 16, width: width - 62, height: height - 52 };
    geometry.points = points;
    geometry.span = points.length > 1
      ? Number(points[points.length - 1].t) - Number(points[0].t)
      : 60;
    var maxValue = 0;
    points.forEach(function (point) {
      var total = series.reduce(function (sum, entry) {
        return sum + (Number(point[entry.key]) || 0);
      }, 0);
      if (total > maxValue) maxValue = total;
    });
    var scaleSpec = {
      maxValue: niceMax(maxValue * 1.08 || 1),
      formatAxis: spec.formatAxis || fmtCompact
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
        var value = Number(point[entry.key]) || 0;
        if (!value) return;
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
      showTooltip(
        context,
        event,
        (spec.tooltipTitle || fmtDate)(point),
        series.map(function (entry) {
          return {
            color: color(entry.color, 'blue'),
            label: entry.label,
            value: (spec.formatValue || fmtNumber)(point[entry.key])
          };
        })
      );
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
      return Math.max(top, Number(item.value) || 0);
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
      fill.style.width = Math.max(1.5, (Number(item.value) / max) * 100) + '%';
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
      return Number(item.value) > 0;
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


  function redrawAll() {
    charts.forEach(function (container) {
      if (container.isConnected && container._lmRender) container._lmRender();
    });
  }

  function register(container) {
    if (!container) return;
    container.setAttribute('data-lm-chart', '1');
    if (charts.indexOf(container) === -1) charts.push(container);
  }

  window.addEventListener('resize', function () {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(redrawAll, 160);
  });

  global.LMCharts = {
    line: function (container, spec) { register(container); lineChart(container, spec); },
    stackedBars: function (container, spec) { register(container); stackedBars(container, spec); },
    bars: function (container, spec) { register(container); hbar(container, spec); },
    donut: function (container, spec) { register(container); donut(container, spec); },
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


