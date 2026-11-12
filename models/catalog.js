'use strict';

/*
 * Catalog workspace behaviour.
 *
 * The data always comes from the catalogue API; the page keeps a light view
 * state (filters, ordering, view mode and the open model) and mirrors it in the
 * URL so a filtered catalog can be bookmarked or shared.
 */
(function () {
  var API = '/api';
  var SEARCH_DELAY = 140;

  /* Natural direction of every sort criterion; the arrow button flips it. */
  var SORTS = {
    catalog: { dir: 'asc' },
    title: { dir: 'asc' },
    category: { dir: 'asc' },
    parameters: { dir: 'desc' },
    methods: { dir: 'desc' }
  };

  var state = {
    models: [],          // catalog summary (model_brief)
    categories: [],      // category index
    total: 0,            // models in the catalog
    category: '',        // active category filter
    tag: '',             // active tag filter (chosen from a card or the drawer)
    attribute: '',       // active availability filter
    query: '',           // active search text
    terms: [],           // normalized search terms (used to highlight)
    sort: 'catalog',
    dir: 'asc',
    view: 'table',
    currentId: null,     // model open in the drawer
    current: null,       // full model payload of the drawer
    requestId: 0,        // discards answers of stale requests
    trigger: null,       // element that opened the drawer (focus returns to it)
    health: null,        // last /api/health payload
    healthChecked: false,// whether /api/health already answered
    loaded: false
  };

  var dom = {};
  var searchTimer = 0;
  var suggestTimer = 0;
  var drawerPressed = false;

  /* ------------------------------------------------------------------ */
  /* Utilities                                                           */
  /* ------------------------------------------------------------------ */

  function byId(id) {
    return document.getElementById(id);
  }

  function each(list, callback) {
    Array.prototype.forEach.call(list || [], callback);
  }

  function closest(element, selector) {
    while (element && element.nodeType === 1) {
      if (element.matches && element.matches(selector)) return element;
      element = element.parentNode;
    }
    return null;
  }

  function isTyping(element) {
    if (!element || element === document.body) return false;
    var tag = element.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
      element.isContentEditable === true;
  }

  function esc(value) {
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' };
    return String(value === null || value === undefined ? '' : value)
      .replace(/[&<>"']/g, function (ch) { return map[ch]; });
  }

  function toast(message, isError) {
    if (!dom.toast) return;
    dom.toast.textContent = message;
    dom.toast.classList.toggle('is-error', Boolean(isError));
    dom.toast.classList.add('is-visible');
    window.clearTimeout(toast.timer);
    toast.timer = window.setTimeout(function () {
      dom.toast.classList.remove('is-visible');
    }, 2600);
  }
  toast.timer = 0;

  function legacyCopy(value, onDone, onFail) {
    var area = document.createElement('textarea');
    area.value = value;
    area.setAttribute('readonly', 'readonly');
    area.setAttribute('aria-hidden', 'true');
    area.style.position = 'fixed';
    area.style.top = '-1000px';
    document.body.appendChild(area);
    var copied = false;
    try {
      area.select();
      copied = document.execCommand('copy');
    } catch (error) {
      copied = false;
    }
    document.body.removeChild(area);
    if (copied) { onDone(); } else { onFail(); }
  }

  function copyText(value, okMessage, failMessage) {
    var text = String(value === null || value === undefined ? '' : value);
    if (!text.trim()) {
      toast(failMessage || 'No hay nada que copiar', true);
      return;
    }
    var done = function () { toast(okMessage || 'Copiado'); };
    var failed = function () { toast(failMessage || 'No se pudo copiar', true); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () {
        legacyCopy(text, done, failed);
      });
      return;
    }
    legacyCopy(text, done, failed);
  }

  function bindCopyButtons(root) {
    each(root.querySelectorAll('[data-copy]'), function (button) {
      button.addEventListener('click', function () {
        var target = document.querySelector(button.getAttribute('data-copy'));
        if (!target) {
          toast('No se encontró el contenido a copiar', true);
          return;
        }
        copyText(target.textContent, button.getAttribute('data-toast') || 'Copiado');
      });
    });
  }

  function escapeRegExp(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  /* Highlight the search terms while escaping every fragment separately, so the
     markup can never be broken by an entity or inject code. */
  function highlight(value, terms) {
    var text = String(value === null || value === undefined ? '' : value);
    var wanted = [];
    each(terms || [], function (term) {
      if (term && term.length > 1 && wanted.indexOf(term) === -1) wanted.push(term);
    });
    if (!wanted.length) return esc(text);
    var pattern = wanted.slice(0, 6).map(escapeRegExp).join('|');
    var matcher;
    try {
      matcher = new RegExp('(' + pattern + ')', 'gi');
    } catch (error) {
      return esc(text);
    }
    var html = '';
    var cursor = 0;
    var match = matcher.exec(text);
    while (match) {
      html += esc(text.slice(cursor, match.index)) + '<mark>' + esc(match[0]) + '</mark>';
      cursor = match.index + match[0].length;
      if (!match[0].length) matcher.lastIndex += 1;
      match = matcher.exec(text);
    }
    return html + esc(text.slice(cursor));
  }

  /* ------------------------------------------------------------------ */
  /* Catalog and service status                                          */
  /* ------------------------------------------------------------------ */

  function loadCatalog() {
    dom.results.setAttribute('aria-busy', 'true');
    window.fetch(API + '/models')
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(function (data) {
        if (!data || !data.models) throw new Error('Respuesta inesperada de la API');
        state.models = data.models;
        state.categories = data.categories || [];
        state.total = typeof data.total === 'number' ? data.total : state.models.length;
        state.loaded = true;
        renderStats();
        renderSidebar();
        renderCategorySelect();
        restoreFromUrl();
      })
      .catch(renderError)
      .then(function () {
        dom.results.removeAttribute('aria-busy');
      });
  }

  function renderError(error) {
    var detail = error && error.message ? error.message : 'Error desconocido';
    dom.results.innerHTML =
      '<div class="notice is-error"><strong>No se pudo cargar el catálogo.</strong>' +
      '<span>' + esc(detail) + '</span>' +
      '<button class="btn btn-secondary btn-sm" type="button" data-action="retry">Reintentar</button></div>';
    dom.resultBar.hidden = true;
  }

  function loadHealth() {
    window.fetch(API + '/health')
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (data) {
        state.health = data || null;
        state.healthChecked = true;
        renderStatus();
        renderStats();
      })
      .catch(function () {
        state.health = null;
        state.healthChecked = true;
        renderStatus();
        renderStats();
      });
  }

  function serviceState() {
    if (!state.healthChecked) {
      return { label: 'Comprobando…', tone: '', dot: 'dot--off' };
    }
    if (!state.health) {
      return { label: 'Sin conexión', tone: 'is-error', dot: 'dot--off' };
    }
    if (state.health.status === 'ok' && state.health.success !== false) {
      return { label: 'Operativo', tone: 'is-ok', dot: 'dot--ok' };
    }
    return { label: 'Degradado', tone: 'is-warn', dot: 'dot--warn' };
  }

  function featureSummary() {
    var health = state.health;
    if (!health) {
      return state.healthChecked ? 'No se pudo leer /api/health' : 'Comprobando el servicio…';
    }
    var parts = [];
    if (health.invoke) {
      parts.push(health.invoke.enabled ? 'Invocación activa' : 'Invocación desactivada');
    }
    if (health.playground) {
      parts.push(health.playground.enabled ? 'Playground activo' : 'Playground desactivado');
    }
    if (health.version) parts.push('v' + health.version);
    return parts.join(' · ') || 'API disponible';
  }

  function renderStatus() {
    if (!dom.serverStatus) return;
    var info = serviceState();
    dom.serverStatus.className = 'status-pill ' + info.tone;
    var dot = dom.serverStatus.querySelector('.dot');
    if (dot) dot.className = 'dot ' + info.dot;
    if (dom.serverStatusText) dom.serverStatusText.textContent = info.label;
    if (dom.sideVersion && state.health && state.health.version) {
      dom.sideVersion.textContent = 'likelihood v' + state.health.version;
    }
  }

  function countWhere(predicate) {
    return state.models.filter(predicate).length;
  }

  function kpiTile(icon, tone, value, label) {
    return '<div class="kpi">' +
      '<span class="kpi-icon ' + tone + '" aria-hidden="true">' + esc(icon) + '</span>' +
      '<div class="kpi-body">' +
      '<div class="kpi-value">' + esc(value) + '</div>' +
      '<div class="kpi-label">' + esc(label) + '</div>' +
      '</div></div>';
  }

  function renderStats() {
    if (!dom.stats) return;
    var info = serviceState();
    var loadable = state.loaded
      ? countWhere(function (model) { return Boolean(model.loadable); })
      : '—';
    var statusTile = '<div class="kpi">' +
      '<span class="kpi-icon is-lava" aria-hidden="true">' +
      '<span class="dot ' + info.dot + '"></span></span>' +
      '<div class="kpi-body">' +
      '<div class="kpi-value is-small">' + esc(info.label) + '</div>' +
      '<div class="kpi-label">' + esc(featureSummary()) + '</div>' +
      '</div></div>';
    dom.stats.innerHTML =
      kpiTile('▦', 'is-blue', state.loaded ? state.total : '—', 'Modelos en el catálogo') +
      kpiTile('◫', '', state.loaded ? state.categories.length : '—', 'Categorías') +
      kpiTile('⇩', 'is-green', loadable, 'Cargables desde archivo') +
      statusTile;
  }

  function renderSidebar() {
    if (!dom.sideCategories) return;
    if (dom.countAll) dom.countAll.textContent = String(state.total);
    if (dom.countLoadable) {
      dom.countLoadable.textContent = String(countWhere(function (model) {
        return Boolean(model.loadable);
      }));
    }
    if (dom.countMethods) {
      dom.countMethods.textContent = String(countWhere(function (model) {
        return (model.method_count || 0) > 0;
      }));
    }
    if (dom.countParameterless) {
      dom.countParameterless.textContent = String(countWhere(function (model) {
        return (model.parameter_count || 0) === 0;
      }));
    }
    dom.sideCategories.innerHTML = state.categories.map(function (category) {
      return '<a class="side-link" href="#" data-category="' + esc(category.id) + '">' +
        '<span class="side-icon" aria-hidden="true">' + esc(category.icon || '•') + '</span>' +
        '<span class="side-label">' + esc(category.label || category.id) + '</span>' +
        '<span class="side-count">' + esc(category.count) + '</span>' +
        '</a>';
    }).join('');
    syncSidebar();
  }

  function syncSidebar() {
    if (!dom.sidebar) return;
    each(dom.sidebar.querySelectorAll('[data-category], [data-attribute], [data-nav]'), function (link) {
      var active = false;
      var category = link.getAttribute('data-category');
      var attribute = link.getAttribute('data-attribute');
      if (link.getAttribute('data-nav') === 'all') {
        active = !state.category && !state.attribute;
      } else if (category !== null) {
        active = Boolean(state.category) && category === state.category;
      } else if (attribute !== null) {
        active = Boolean(state.attribute) && attribute === state.attribute;
      }
      link.classList.toggle('is-active', active);
      if (active) {
        link.setAttribute('aria-current', 'page');
      } else {
        link.removeAttribute('aria-current');
      }
    });
  }

  function renderCategorySelect() {
    if (!dom.categorySelect) return;
    var options = ['<option value="">Todas las categorías</option>'];
    state.categories.forEach(function (category) {
      options.push('<option value="' + esc(category.id) + '">' +
        esc((category.icon ? category.icon + ' ' : '') + category.label) +
        ' (' + esc(category.count) + ')</option>');
    });
    dom.categorySelect.innerHTML = options.join('');
    dom.categorySelect.value = state.category;
  }

  /* ------------------------------------------------------------------ */
  /* Filtering and ordering                                              */
  /* ------------------------------------------------------------------ */

  function attributeMatches(model) {
    if (state.attribute === 'loadable') return Boolean(model.loadable);
    if (state.attribute === 'methods') return (model.method_count || 0) > 0;
    if (state.attribute === 'parameterless') return (model.parameter_count || 0) === 0;
    return true;
  }

  function hasTag(model, tag) {
    var wanted = String(tag || '').toLowerCase();
    if (!wanted) return true;
    return (model.tags || []).some(function (item) {
      return String(item).toLowerCase() === wanted;
    });
  }

  function searchHaystack(model) {
    return [
      model.id, model.title, model.category, model.category_label, model.module,
      model.signature, model.import_statement || '', model.short_description || '',
      (model.tags || []).join(' ')
    ].join(' ').toLowerCase();
  }

  function matches(model, terms) {
    if (state.category && model.category !== state.category) return false;
    if (state.tag && !hasTag(model, state.tag)) return false;
    if (!attributeMatches(model)) return false;
    if (!terms.length) return true;
    var haystack = searchHaystack(model);
    for (var i = 0; i < terms.length; i += 1) {
      if (haystack.indexOf(terms[i]) === -1) return false;
    }
    return true;
  }

  /* Relevance: the id and the title weigh more than the description. */
  function scoreModel(model, terms, needle) {
    var score = 0;
    var id = String(model.id || '').toLowerCase();
    var title = String(model.title || '').toLowerCase();
    var description = String(model.short_description || '').toLowerCase();
    var signature = String(model.signature || '').toLowerCase();
    if (id === needle) score += 120;
    if (title === needle) score += 100;
    if (id.indexOf(needle) === 0) score += 60;
    else if (id.indexOf(needle) > 0) score += 25;
    if (title.indexOf(needle) === 0) score += 50;
    else if (title.indexOf(needle) > 0) score += 20;
    each(model.tags || [], function (tag) {
      var lowered = String(tag).toLowerCase();
      for (var i = 0; i < terms.length; i += 1) {
        if (lowered === terms[i]) score += 30;
        else if (lowered.indexOf(terms[i]) === 0) score += 12;
      }
    });
    for (var index = 0; index < terms.length; index += 1) {
      if (description.indexOf(terms[index]) !== -1) score += 6;
      if (signature.indexOf(terms[index]) !== -1) score += 4;
    }
    return score;
  }

  function naturalDir(key) {
    return SORTS[key] ? SORTS[key].dir : 'asc';
  }

  function compareModels(a, b) {
    var primary = 0;
    if (state.sort === 'title') {
      primary = a.title.localeCompare(b.title, 'es');
    } else if (state.sort === 'category') {
      primary = String(a.category_label || '').localeCompare(String(b.category_label || ''), 'es');
    } else if (state.sort === 'parameters') {
      primary = (a.parameter_count || 0) - (b.parameter_count || 0);
    } else if (state.sort === 'methods') {
      primary = (a.method_count || 0) - (b.method_count || 0);
    }
    if (primary !== 0) return state.dir === 'desc' ? -primary : primary;
    return a.title.localeCompare(b.title, 'es');
  }

  function sortModels(list) {
    var sorted = list.slice();
    if (state.sort === 'catalog') {
      return state.dir === 'desc' ? sorted.reverse() : sorted;
    }
    return sorted.sort(compareModels);
  }

  function visibleModels() {
    var needle = state.query.trim().toLowerCase();
    var terms = needle ? needle.split(/\s+/) : [];
    state.terms = terms;
    var filtered = state.models.filter(function (model) {
      return matches(model, terms);
    });
    if (!terms.length) return sortModels(filtered);
    var scored = filtered.map(function (model) {
      return { model: model, score: scoreModel(model, terms, needle) };
    });
    scored.sort(function (a, b) {
      return (b.score - a.score) || compareModels(a.model, b.model);
    });
    return scored.map(function (entry) { return entry.model; });
  }

  function applyFilters() {
    var list = visibleModels();
    renderResults(list);
    renderResultBar(list.length);
    renderSortDirection();
    syncSidebar();
    syncViewButtons();
    syncUrl(false);
  }

  /* ------------------------------------------------------------------ */
  /* Rendering                                                           */
  /* ------------------------------------------------------------------ */

  function previewUrl(model) {
    return '/preview/' + encodeURIComponent(model.id);
  }

  function notebookUrl(model) {
    return '/api/notebook/' + encodeURIComponent(model.id);
  }

  function modelTags(model, limit) {
    var terms = state.terms || [];
    var tags = (model.tags || []).slice(0, limit);
    if (!tags.length) return '<span class="faint">—</span>';
    return '<div class="tag-row">' + tags.map(function (tag) {
      var active = state.tag && String(tag).toLowerCase() === state.tag.toLowerCase();
      return '<button type="button" class="tag' + (active ? ' is-active' : '') +
        '" data-tag="' + esc(tag) + '" aria-pressed="' + (active ? 'true' : 'false') +
        '" title="Filtrar por la etiqueta ' + esc(tag) + '">' + highlight(tag, terms) + '</button>';
    }).join('') + '</div>';
  }

  function sortHeader(key, label, extraClass) {
    var active = state.sort === key;
    var mark = active ? (state.dir === 'asc' ? '▲' : '▼') : '';
    var classes = 'sortable' + (extraClass ? ' ' + extraClass : '');
    return '<th scope="col" class="' + classes + '"' +
      (active ? ' aria-sort="' + (state.dir === 'asc' ? 'ascending' : 'descending') + '"' : '') +
      '><button type="button" class="sort-btn" data-sort="' + key + '">' + esc(label) +
      (mark ? '<span class="sort-mark" aria-hidden="true">' + mark + '</span>' : '') +
      '</button></th>';
  }

  function tableRow(model, terms) {
    var url = previewUrl(model);
    var description = String(model.short_description || '');
    if (description.length > 180) description = description.slice(0, 180) + '…';
    var availability = model.loadable
      ? '<span class="badge badge-green"><span class="dot"></span>Cargable</span>'
      : '<span class="faint">—</span>';
    return '<tr data-id="' + esc(model.id) + '"' +
      (state.currentId === model.id ? ' class="is-selected"' : '') + '>' +
      '<td><div class="cell-name">' +
      '<span class="model-icon" aria-hidden="true">' + esc(model.icon) + '</span>' +
      '<div><a class="model-title" href="' + url + '" data-action="open">' +
      highlight(model.title, terms) + '</a>' +
      '<span class="model-id">' + esc(model.id) + '</span></div></div></td>' +
      '<td><span class="badge category-' + esc(model.category) + '">' +
      esc(model.category_label) + '</span></td>' +
      '<td class="cell-desc hide-md">' + highlight(description, terms) + '</td>' +
      '<td class="cell-num">' + esc(model.parameter_count || 0) + '</td>' +
      '<td class="cell-num">' + esc(model.method_count || 0) + '</td>' +
      '<td class="hide-sm">' + availability + '</td>' +
      '<td class="hide-md">' + modelTags(model, 2) + '</td>' +
      '<td class="cell-actions">' +
      '<button class="btn btn-secondary btn-sm" type="button" data-action="open">Ficha</button>' +
      '<a class="btn btn-ghost btn-sm hide-sm" href="' + url +
      '" target="_blank" rel="noopener">Vista previa</a>' +
      '<a class="btn btn-ghost btn-sm hide-sm" href="' + notebookUrl(model) +
      '" download>Notebook</a>' +
      '</td></tr>';
  }

  function renderTable(list) {
    var terms = state.terms || [];
    return '<div class="table-wrap"><table class="table">' +
      '<caption class="sr-only">Modelos del catálogo</caption>' +
      '<thead><tr>' +
      sortHeader('title', 'Modelo') +
      sortHeader('category', 'Categoría') +
      '<th scope="col" class="hide-md">Descripción</th>' +
      sortHeader('parameters', 'Parámetros', 'num-head') +
      sortHeader('methods', 'Métodos', 'num-head') +
      '<th scope="col" class="hide-sm">Carga</th>' +
      '<th scope="col" class="hide-md">Etiquetas</th>' +
      '<th scope="col" class="cell-actions"><span class="sr-only">Acciones</span></th>' +
      '</tr></thead><tbody>' +
      list.map(function (model) { return tableRow(model, terms); }).join('') +
      '</tbody></table></div>';
  }

  function renderEmpty() {
    var detail = state.query
      ? 'Ningún modelo coincide con «' + state.query + '»' +
        (state.tag ? ' y la etiqueta «' + state.tag + '»' : '') + '.'
      : 'No hay modelos con esta combinación de filtros.';
    dom.results.innerHTML =
      '<div class="empty-state">' +
      '<span class="empty-icon" aria-hidden="true">⌕</span>' +
      '<h3>Sin resultados</h3>' +
      '<p>' + esc(detail) + ' Prueba con menos palabras, otra categoría o quita las etiquetas activas.</p>' +
      '<button class="btn btn-secondary btn-sm" type="button" data-action="reset">Restablecer filtros</button>' +
      '</div>';
  }

  function renderResults(list) {
    if (!list.length) {
      renderEmpty();
      return;
    }
    dom.results.innerHTML = state.view === 'cards' ? renderCards(list) : renderTable(list);
  }

  var SORT_LABELS = {
    catalog: 'orden del catálogo',
    title: 'nombre',
    category: 'categoría',
    parameters: 'parámetros',
    methods: 'métodos'
  };

  function cardHtml(model) {
    var terms = state.terms || [];
    var description = String(model.short_description || '');
    if (description.length > 200) description = description.slice(0, 200) + '…';
    var url = previewUrl(model);
    var availability = model.loadable
      ? '<span class="meta-sep">·</span><span class="meta-item">' +
        '<span class="dot dot--ok"></span>Cargable</span>'
      : '';
    return '<article class="catalog-card' +
      (state.currentId === model.id ? ' is-selected' : '') +
      '" data-id="' + esc(model.id) + '">' +
      '<div class="catalog-card-head">' +
      '<span class="catalog-card-icon" aria-hidden="true">' + esc(model.icon) + '</span>' +
      '<div class="catalog-card-heading">' +
      '<a class="catalog-card-title" href="' + url + '" data-action="open">' +
      highlight(model.title, terms) + '</a>' +
      '<span class="catalog-card-id">' + esc(model.id) + '</span>' +
      '</div>' +
      '<span class="badge category-' + esc(model.category) + '">' +
      esc(model.category_label) + '</span>' +
      '</div>' +
      '<p class="catalog-card-desc clamp-3">' + highlight(description, terms) + '</p>' +
      modelTags(model, 4) +
      '<div class="catalog-card-meta">' +
      '<span class="meta-item">' + esc(model.parameter_count || 0) + ' parámetros</span>' +
      '<span class="meta-sep">·</span>' +
      '<span class="meta-item">' + esc(model.method_count || 0) + ' métodos</span>' +
      availability +
      '</div>' +
      '<div class="catalog-card-actions">' +
      '<button class="btn btn-secondary btn-sm" type="button" data-action="open">Ficha</button>' +
      '<a class="btn btn-ghost btn-sm" href="' + url + '" target="_blank" rel="noopener">Vista previa</a>' +
      '<a class="btn btn-ghost btn-sm" href="' + notebookUrl(model) + '" download>Notebook</a>' +
      '</div></article>';
  }

  function renderCards(list) {
    return '<div class="card-grid">' +
      list.map(cardHtml).join('') +
      '</div>';
  }

  function categoryLabel(id) {
    for (var index = 0; index < state.categories.length; index += 1) {
      if (state.categories[index].id === id) {
        var category = state.categories[index];
        return (category.icon ? category.icon + ' ' : '') + (category.label || category.id);
      }
    }
    return id;
  }

  function attributeLabel(id) {
    var labels = {
      loadable: 'Cargables desde archivo',
      methods: 'Con métodos públicos',
      parameterless: 'Sin parámetros'
    };
    return labels[id] || id;
  }

  function filterChip(kind, label) {
    return '<button type="button" class="chip" data-clear="' + kind +
      '" title="Quitar este filtro">' + esc(label) +
      '<span class="chip-x" aria-hidden="true">×</span>' +
      '<span class="sr-only">Quitar el filtro</span></button>';
  }

  function renderResultBar(count) {
    if (!dom.resultBar) return;
    var noun = count === 1 ? 'modelo' : 'modelos';
    var text = count === state.total
      ? 'Mostrando los ' + count + ' ' + noun + ' del catálogo'
      : 'Mostrando ' + count + ' de ' + state.total + ' ' + noun;
    dom.resultCount.textContent = text;
    dom.resultBar.hidden = false;
    if (dom.resultHint) {
      dom.resultHint.textContent = state.terms.length
        ? 'Ordenados por relevancia'
        : 'Orden: ' + (SORT_LABELS[state.sort] || state.sort) +
          (state.dir === 'asc' ? ' ↑' : ' ↓');
    }
    var chips = [];
    if (state.category) chips.push(filterChip('category', 'Categoría: ' + categoryLabel(state.category)));
    if (state.attribute) chips.push(filterChip('attribute', 'Atributo: ' + attributeLabel(state.attribute)));
    if (state.tag) chips.push(filterChip('tag', 'Etiqueta: ' + state.tag));
    if (state.query) chips.push(filterChip('query', 'Búsqueda: ' + state.query));
    dom.activeFilters.innerHTML = chips.length
      ? '<span class="chips-label">Filtros</span>' + chips.join('')
      : '';
    dom.activeFilters.hidden = !chips.length;
    if (dom.reset) dom.reset.hidden = !chips.length;
  }

  function renderSortDirection() {
    if (!dom.sortDir) return;
    var descending = state.dir === 'desc';
    var numeric = state.sort === 'parameters' || state.sort === 'methods';
    var hint = descending
      ? (numeric ? 'de menos a más' : 'de la Z a la A')
      : (numeric ? 'de más a menos' : 'de la A a la Z');
    if (state.sort === 'catalog') {
      hint = descending ? 'del último al primero' : 'del primero al último';
    }
    dom.sortDir.textContent = descending ? '↓' : '↑';
    dom.sortDir.title = 'Invertir el orden (' + hint + ')';
    dom.sortDir.setAttribute('aria-label', 'Invertir el orden, ahora ' + hint);
  }

  function syncViewButtons() {
    each([dom.viewTable, dom.viewCards], function (button) {
      if (!button) return;
      button.setAttribute(
        'aria-pressed',
        button.getAttribute('data-view') === state.view ? 'true' : 'false'
      );
    });
  }

  /* ------------------------------------------------------------------ */
  /* Model details drawer                                                */
  /* ------------------------------------------------------------------ */

  function findModel(id) {
    for (var index = 0; index < state.models.length; index += 1) {
      if (state.models[index].id === id) return state.models[index];
    }
    return null;
  }

  function highlightSelectedRow() {
    each(dom.results.querySelectorAll('[data-id]'), function (node) {
      node.classList.toggle('is-selected', node.getAttribute('data-id') === state.currentId);
    });
  }

  function modelLoadable(model) {
    if (typeof model.loadable === 'boolean') return model.loadable;
    if (model.load && typeof model.load.loadable === 'boolean') return model.load.loadable;
    return Boolean((model.loaders || []).length);
  }

  function setDrawerHeader(model) {
    var availability = modelLoadable(model)
      ? '<span class="badge badge-green">cargable</span>' : '';
    dom.drawerEyebrow.innerHTML =
      '<span class="badge category-' + esc(model.category) + '">' +
      esc(model.category_label) + '</span>' +
      '<span class="mono">' + esc(model.id) + '</span>' +
      availability;
    dom.drawerTitle.innerHTML =
      '<span class="drawer-icon" aria-hidden="true">' + esc(model.icon) + '</span>' +
      esc(model.title || model.id);
    dom.drawerSubtitle.textContent = model.short_description || '';
  }

  function openDrawer(id, trigger, push) {
    var summary = findModel(id);
    if (!summary) {
      toast('Modelo no encontrado: ' + id, true);
      return;
    }
    state.currentId = id;
    state.current = summary;
    state.trigger = trigger || null;
    setDrawerHeader(summary);
    dom.drawerBody.innerHTML =
      '<div class="loader"><span class="spinner" aria-hidden="true"></span>' +
      'Cargando la ficha del modelo…</div>';
    dom.drawerFooter.innerHTML = '';
    dom.drawerBackdrop.classList.add('is-open');
    document.body.classList.add('db-body-locked');
    highlightSelectedRow();
    if (push !== false) syncUrl(true);
    dom.drawerClose.focus();

    var requestId = (state.requestId += 1);
    window.fetch(API + '/models/' + encodeURIComponent(id))
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(function (data) {
        if (requestId !== state.requestId) return;
        if (!data || !data.success || !data.model) throw new Error('Modelo no disponible');
        renderDrawer(data.model);
      })
      .catch(function (error) {
        if (requestId !== state.requestId) return;
        dom.drawerBody.innerHTML =
          '<div class="notice is-error"><strong>No se pudo cargar la ficha.</strong>' +
          '<span>' + esc(error && error.message ? error.message : 'Error desconocido') + '</span></div>';
      });
  }

  function closeDrawer(options) {
    var settings = options || {};
    if (!dom.drawerBackdrop.classList.contains('is-open')) return;
    state.requestId += 1;
    dom.drawerBackdrop.classList.remove('is-open');
    document.body.classList.remove('db-body-locked');
    state.currentId = null;
    state.current = null;
    if (settings.push !== false) syncUrl(true);
    highlightSelectedRow();
    if (state.trigger && document.contains(state.trigger)) state.trigger.focus();
    state.trigger = null;
  }

  function metaBox(label, value, options) {
    var settings = options || {};
    return '<div class="meta-item-box">' +
      '<span class="meta-label">' + esc(label) + '</span>' +
      '<span class="meta-value' + (settings.mono === false ? '' : ' mono') + '"' +
      (settings.id ? ' id="' + settings.id + '"' : '') + '>' +
      (settings.raw ? value : esc(value)) +
      '</span></div>';
  }

  function metaBoxes(model) {
    var loaders = (model.loaders || []).map(function (loader) { return loader.name; });
    var boxes = [
      metaBox('Módulo', model.module || '—'),
      metaBox('Firma', model.signature || '—'),
      metaBox('Importación', model.import_statement || '—', { id: 'drawer-import' }),
      metaBox('Archivo', (model.path || '—') + ':' + (model.lineno || '—')),
      metaBox('Líneas de código', model.line_count || 0, { mono: false }),
      metaBox('Métodos invocables', (model.invokable_methods || []).length, { mono: false }),
      metaBox(
        'Carga desde archivo',
        modelLoadable(model) ? 'Disponible' : 'No declarada',
        { mono: false }
      )
    ];
    if (loaders.length) boxes.push(metaBox('Loaders', loaders.join(', ')));
    return boxes.join('');
  }

  function parametersSection(model) {
    var parameters = model.parameters || [];
    if (!parameters.length) {
      return '<p class="empty">Este modelo no declara parámetros de entrada.</p>';
    }
    return '<div class="grid">' + parameters.map(function (parameter) {
      var fallback = (parameter.default === null || parameter.default === undefined)
        ? 'requerido' : parameter.default;
      var required = parameter.required
        ? ' <span class="badge badge-lava">requerido</span>' : '';
      return '<div class="param-card">' +
        '<div class="param-name">' + esc(parameter.name) + required + '</div>' +
        '<div class="param-type">' + esc(parameter.type || 'sin tipo') +
        ' · ' + esc(fallback) + '</div>' +
        '<div class="param-desc">' + esc(parameter.description || '—') + '</div>' +
        '</div>';
    }).join('') + '</div>';
  }

  function attributesSection(model) {
    var attributes = model.attributes || [];
    if (!attributes.length) return '';
    return '<section class="drawer-section"><h3>Atributos</h3><div class="grid">' +
      attributes.map(function (attribute) {
        return '<div class="param-card">' +
          '<div class="param-name">' + esc(attribute.name) + '</div>' +
          '<div class="param-type">' + esc(attribute.type || '—') + '</div>' +
          '<div class="param-desc">' + esc(attribute.description || '—') + '</div>' +
          '</div>';
      }).join('') + '</div></section>';
  }

  function methodsSection(model) {
    var methods = model.methods || [];
    var invokable = model.invokable_methods || [];
    if (!methods.length) {
      return '<p class="empty">Este modelo no expone métodos públicos.</p>';
    }
    return '<ul class="methods">' + methods.map(function (method) {
      var signature = method.signature || method.name || '';
      var badge = invokable.indexOf(method.name) !== -1
        ? '<span class="badge badge-blue">invocable</span>' : '';
      return '<li>' +
        '<span class="row"><span class="method-name" title="' + esc(signature) + '">' +
        esc(signature.split(' -> ')[0]) + '</span>' +
        '<span class="method-origin">' + (method.inherited ? 'heredado' : 'propio') + '</span>' +
        badge + '</span>' +
        '<span class="method-desc">' + esc(method.description || '—') + '</span>' +
        '</li>';
    }).join('') + '</ul>';
  }

  function exampleSection(model) {
    return '<section class="drawer-section"><h3>Ejemplo de uso</h3>' +
      '<div class="code-head"><span class="code-label">Python</span>' +
      '<button class="btn btn-ghost btn-xs" type="button" data-copy="#drawer-example"' +
      ' data-toast="Ejemplo copiado">Copiar</button></div>' +
      '<pre class="code" id="drawer-example">' + esc(model.example || '# —') + '</pre>' +
      '</section>';
  }

  function renderDrawer(model) {
    state.current = model;
    setDrawerHeader(model);
    var sections = [
      '<section class="drawer-section"><h3>Descripción</h3><div class="prose">' +
      esc(model.full_description || model.short_description || '') + '</div></section>',
      '<section class="drawer-section"><h3>Metadatos</h3><div class="meta-grid">' +
      metaBoxes(model) + '</div></section>',
      '<section class="drawer-section"><h3>Parámetros' +
      ' <span class="panel-sub">' + (model.parameters || []).length + '</span></h3>' +
      parametersSection(model) + '</section>',
      attributesSection(model),
      '<section class="drawer-section"><h3>Métodos' +
      ' <span class="panel-sub">' + (model.methods || []).length + '</span></h3>' +
      methodsSection(model) + '</section>',
      exampleSection(model)
    ];
    dom.drawerBody.innerHTML = sections.join('');
    dom.drawerFooter.innerHTML =
      '<a class="btn btn-primary" href="' + previewUrl(model) + '">Abrir página del modelo</a>' +
      '<a class="btn btn-secondary" href="' + notebookUrl(model) + '" download>Notebook</a>' +
      '<button class="btn btn-secondary" type="button" data-copy="#drawer-import"' +
      ' data-toast="Importación copiada">Copiar import</button>' +
      '<a class="btn btn-ghost" href="/api/models/' + encodeURIComponent(model.id) +
      '/source" target="_blank" rel="noopener">Código fuente</a>';
    bindCopyButtons(dom.drawerBody);
    bindCopyButtons(dom.drawerFooter);
  }

  /* ------------------------------------------------------------------ */
  /* URL state                                                           */
  /* ------------------------------------------------------------------ */

  function readQuery() {
    var result = {
      category: '', attribute: '', tag: '', q: '',
      sort: 'catalog', dir: '', view: 'table', model: ''
    };
    var raw = window.location.search.replace(/^\?/, '');
    if (!raw) return result;
    each(raw.split('&'), function (pair) {
      if (!pair) return;
      var index = pair.indexOf('=');
      var key = decodeURIComponent(index === -1 ? pair : pair.slice(0, index));
      var value = index === -1 ? '' : decodeURIComponent(pair.slice(index + 1).replace(/\+/g, ' '));
      if (key === 'category') result.category = value;
      else if (key === 'tag' || key === 'tags') result.tag = value;
      else if (key === 'q') result.q = value;
      else if (key === 'sort') result.sort = value;
      else if (key === 'dir' || key === 'order') result.dir = value;
      else if (key === 'view') result.view = value;
      else if (key === 'attribute' || key === 'attr') result.attribute = value;
      else if (key === 'model') result.model = value;
    });
    return result;
  }

  function syncUrl(push) {
    if (!window.history || !window.history.pushState) return;
    var params = [];
    if (state.category) params.push('category=' + encodeURIComponent(state.category));
    if (state.attribute) params.push('attribute=' + encodeURIComponent(state.attribute));
    if (state.tag) params.push('tag=' + encodeURIComponent(state.tag));
    if (state.query) params.push('q=' + encodeURIComponent(state.query));
    if (state.sort !== 'catalog') params.push('sort=' + encodeURIComponent(state.sort));
    if (state.dir !== naturalDir(state.sort)) params.push('dir=' + state.dir);
    if (state.view !== 'table') params.push('view=' + state.view);
    if (state.currentId) params.push('model=' + encodeURIComponent(state.currentId));
    var url = window.location.pathname + (params.length ? '?' + params.join('&') : '');
    if (push) {
      window.history.pushState({ catalog: true }, '', url);
    } else {
      window.history.replaceState({ catalog: true }, '', url);
    }
  }

  function applyQuery(query) {
    state.category = query.category || '';
    state.attribute = query.attribute || '';
    state.tag = query.tag || '';
    state.query = query.q || '';
    state.sort = SORTS[query.sort] ? query.sort : 'catalog';
    state.dir = query.dir === 'asc' || query.dir === 'desc' ? query.dir : naturalDir(state.sort);
    state.view = query.view === 'cards' ? 'cards' : 'table';
    dom.search.value = state.query;
    dom.clear.hidden = !state.query;
    dom.sort.value = state.sort;
    renderCategorySelect();
    applyFilters();
    if (query.model && findModel(query.model)) {
      openDrawer(query.model, null, false);
    }
  }

  function restoreFromUrl() {
    applyQuery(readQuery());
  }

  /* ------------------------------------------------------------------ */
  /* Filter mutations                                                    */
  /* ------------------------------------------------------------------ */

  function setCategory(category) {
    state.category = state.category === category ? '' : category;
    renderCategorySelect();
    applyFilters();
  }

  function setAttribute(attribute) {
    state.attribute = state.attribute === attribute ? '' : attribute;
    applyFilters();
  }

  function setTag(tag) {
    var value = String(tag || '').trim();
    state.tag = state.tag.toLowerCase() === value.toLowerCase() ? '' : value;
    applyFilters();
  }

  function setSort(key) {
    if (!SORTS[key]) return;
    if (state.sort === key) {
      state.dir = state.dir === 'asc' ? 'desc' : 'asc';
    } else {
      state.sort = key;
      state.dir = naturalDir(key);
    }
    dom.sort.value = state.sort;
    applyFilters();
  }

  function setView(view) {
    state.view = view === 'cards' ? 'cards' : 'table';
    applyFilters();
  }

  function clearFilter(kind) {
    if (kind === 'category') {
      state.category = '';
      renderCategorySelect();
    } else if (kind === 'attribute') {
      state.attribute = '';
    } else if (kind === 'tag') {
      state.tag = '';
    } else if (kind === 'query') {
      state.query = '';
      dom.search.value = '';
      dom.clear.hidden = true;
    }
    applyFilters();
  }

  function resetFilters() {
    state.category = '';
    state.attribute = '';
    state.tag = '';
    state.query = '';
    dom.search.value = '';
    dom.clear.hidden = true;
    renderCategorySelect();
    applyFilters();
    if (dom.search) dom.search.focus();
    toast('Filtros restablecidos');
  }

  /* ------------------------------------------------------------------ */
  /* Chrome: autocomplete, sidebar, theme                                */
  /* ------------------------------------------------------------------ */

  function loadSuggestions(value) {
    window.clearTimeout(suggestTimer);
    if (!dom.suggestions) return;
    if (value.length < 2) {
      dom.suggestions.innerHTML = '';
      return;
    }
    suggestTimer = window.setTimeout(function () {
      window.fetch(API + '/search?q=' + encodeURIComponent(value) + '&limit=6')
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (data) {
          if (!data || !data.suggestions) return;
          dom.suggestions.innerHTML = data.suggestions.map(function (item) {
            return '<option value="' + esc(item.title) + '" label="' +
              esc(item.title + ' · ' + item.category_label) + '"></option>';
          }).join('');
        })
        .catch(function () { /* the autocomplete is optional */ });
    }, 250);
  }

  function isMobileLayout() {
    return window.matchMedia('(max-width: 900px)').matches;
  }

  function syncSidebarToggle() {
    var mobile = isMobileLayout();
    var expanded = mobile
      ? document.body.classList.contains('sidebar-open')
      : !document.body.classList.contains('sidebar-collapsed');
    if (dom.sidebarToggle) {
      dom.sidebarToggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }
    if (dom.sidebarOverlay) {
      dom.sidebarOverlay.hidden = !(mobile && document.body.classList.contains('sidebar-open'));
    }
  }

  function toggleSidebar() {
    if (isMobileLayout()) {
      document.body.classList.toggle('sidebar-open');
    } else {
      document.body.classList.toggle('sidebar-collapsed');
    }
    syncSidebarToggle();
  }

  function closeMobileSidebar() {
    document.body.classList.remove('sidebar-open');
    syncSidebarToggle();
  }

  function currentTheme() {
    var explicit = document.documentElement.getAttribute('data-theme');
    if (explicit === 'light' || explicit === 'dark') return explicit;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function toggleTheme() {
    var next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try {
      window.localStorage.setItem('lm-theme', next);
    } catch (error) {
      /* storage is optional */
    }
    toast(next === 'dark' ? 'Tema oscuro activado' : 'Tema claro activado');
  }

  function renderSkeleton() {
    var rows = '';
    for (var index = 0; index < 7; index += 1) {
      rows += '<div class="skeleton skeleton-row"></div>';
    }
    dom.results.innerHTML = '<div class="skeleton-list">' + rows + '</div>';
  }

  /* ------------------------------------------------------------------ */
  /* Events                                                              */
  /* ------------------------------------------------------------------ */

  function onSearchInput() {
    window.clearTimeout(searchTimer);
    var value = dom.search.value.trim();
    dom.clear.hidden = !value;
    loadSuggestions(value);
    searchTimer = window.setTimeout(function () {
      if (value === state.query) return;
      state.query = value;
      applyFilters();
    }, SEARCH_DELAY);
  }

  function trapDrawerFocus(event) {
    var focusable = dom.drawer.querySelectorAll(
      'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (!focusable.length) return;
    var first = focusable[0];
    var last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function onResultsClick(event) {
    var target = event.target;
    if (!target || !target.closest) return;
    if (closest(target, '[data-action="retry"]')) {
      loadCatalog();
      return;
    }
    if (closest(target, '[data-action="reset"]')) {
      resetFilters();
      return;
    }
    var tagButton = closest(target, '[data-tag]');
    if (tagButton) {
      setTag(tagButton.getAttribute('data-tag'));
      return;
    }
    var sortButton = closest(target, '[data-sort]');
    if (sortButton) {
      setSort(sortButton.getAttribute('data-sort'));
      return;
    }
    var row = closest(target, '[data-id]');
    if (!row) return;
    var openNode = closest(target, '[data-action="open"]');
    var control = closest(target, 'a, button');
    /* Links and buttons such as "Vista previa" or "Notebook" act on their own. */
    if (control && !openNode) return;
    if (openNode && openNode.tagName === 'A') {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
        return;
      }
      event.preventDefault();
    }
    openDrawer(row.getAttribute('data-id'), openNode || row);
  }

  function onDocumentKeydown(event) {
    if (dom.drawerBackdrop.classList.contains('is-open')) {
      if (event.key === 'Escape' || event.key === 'Esc') {
        event.preventDefault();
        closeDrawer();
        return;
      }
      if (event.key === 'Tab') {
        trapDrawerFocus(event);
        return;
      }
    }
    if (event.key === 'Escape' || event.key === 'Esc') {
      if (document.body.classList.contains('sidebar-open')) {
        closeMobileSidebar();
        return;
      }
      if (dom.search.value) {
        dom.search.value = '';
        dom.clear.hidden = true;
        onSearchInput();
      }
      return;
    }
    if ((event.key === 'k' || event.key === 'K') && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      dom.search.focus();
      dom.search.select();
      return;
    }
    if (event.key === '/' && !isTyping(event.target)) {
      event.preventDefault();
      dom.search.focus();
      dom.search.select();
    }
  }

  function bindEvents() {
    dom.sidebarToggle.addEventListener('click', toggleSidebar);
    dom.sidebarOverlay.addEventListener('click', closeMobileSidebar);
    dom.themeToggle.addEventListener('click', toggleTheme);

    dom.search.addEventListener('input', onSearchInput);
    dom.clear.addEventListener('click', function () {
      dom.search.value = '';
      dom.clear.hidden = true;
      onSearchInput();
      dom.search.focus();
    });
    dom.categorySelect.addEventListener('change', function () {
      state.category = dom.categorySelect.value;
      applyFilters();
    });
    dom.sort.addEventListener('change', function () {
      state.sort = SORTS[dom.sort.value] ? dom.sort.value : 'catalog';
      state.dir = naturalDir(state.sort);
      applyFilters();
    });
    dom.sortDir.addEventListener('click', function () {
      state.dir = state.dir === 'asc' ? 'desc' : 'asc';
      applyFilters();
    });
    dom.viewTable.addEventListener('click', function () { setView('table'); });
    dom.viewCards.addEventListener('click', function () { setView('cards'); });
    dom.reset.addEventListener('click', resetFilters);

    dom.activeFilters.addEventListener('click', function (event) {
      var button = closest(event.target, '[data-clear]');
      if (button) clearFilter(button.getAttribute('data-clear'));
    });

    dom.sidebar.addEventListener('click', function (event) {
      var link = closest(event.target, '[data-category], [data-attribute], [data-nav]');
      if (!link) return;
      event.preventDefault();
      if (link.getAttribute('data-nav') === 'all') {
        state.category = '';
        state.attribute = '';
        renderCategorySelect();
        applyFilters();
      } else if (link.hasAttribute('data-category')) {
        setCategory(link.getAttribute('data-category'));
      } else {
        setAttribute(link.getAttribute('data-attribute'));
      }
      if (isMobileLayout()) closeMobileSidebar();
    });

    dom.results.addEventListener('click', onResultsClick);

    dom.drawerClose.addEventListener('click', function () { closeDrawer(); });
    dom.drawerBackdrop.addEventListener('mousedown', function (event) {
      drawerPressed = event.target === dom.drawerBackdrop;
    });
    dom.drawerBackdrop.addEventListener('click', function (event) {
      if (event.target === dom.drawerBackdrop && drawerPressed) closeDrawer();
      drawerPressed = false;
    });

    document.addEventListener('keydown', onDocumentKeydown);
    window.addEventListener('popstate', restoreFromUrl);
    window.addEventListener('resize', function () {
      if (!isMobileLayout()) closeMobileSidebar();
      syncSidebarToggle();
    });
  }

  /* ------------------------------------------------------------------ */
  /* Bootstrap                                                           */
  /* ------------------------------------------------------------------ */

  function init() {
    dom.results = byId('results');
    dom.stats = byId('stats');
    dom.activeFilters = byId('active-filters');
    dom.search = byId('search');
    dom.suggestions = byId('search-suggestions');
    dom.clear = byId('clear-search');
    dom.categorySelect = byId('category-select');
    dom.sort = byId('sort');
    dom.sortDir = byId('sort-dir');
    dom.viewTable = byId('view-table');
    dom.viewCards = byId('view-cards');
    dom.reset = byId('reset-filters');
    dom.resultBar = byId('result-bar');
    dom.resultCount = byId('result-count');
    dom.resultHint = byId('result-hint');
    dom.sidebar = byId('sidebar');
    dom.sideCategories = byId('side-categories');
    dom.sidebarToggle = byId('sidebar-toggle');
    dom.sidebarOverlay = byId('sidebar-overlay');
    dom.countAll = byId('count-all');
    dom.countLoadable = byId('count-loadable');
    dom.countMethods = byId('count-methods');
    dom.countParameterless = byId('count-parameterless');
    dom.sideVersion = byId('side-version');
    dom.serverStatus = byId('server-status');
    dom.serverStatusText = byId('server-status-text');
    dom.themeToggle = byId('theme-toggle');
    dom.toast = byId('toast');
    dom.drawerBackdrop = byId('drawer-backdrop');
    dom.drawer = byId('drawer');
    dom.drawerTitle = byId('drawer-title');
    dom.drawerSubtitle = byId('drawer-subtitle');
    dom.drawerEyebrow = byId('drawer-eyebrow');
    dom.drawerBody = byId('drawer-body');
    dom.drawerFooter = byId('drawer-footer');
    dom.drawerClose = byId('drawer-close');

    renderSkeleton();
    renderStats();
    renderStatus();
    syncViewButtons();
    syncSidebarToggle();
    bindEvents();
    loadHealth();
    loadCatalog();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();











