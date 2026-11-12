import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const protectedFiles = new Set([
  '.gitignore', 'README.md',
  'scripts/benchmark_dual_source.py', 'scripts/benchmark_speaker_separation.py',
]);
const privateRoots = /^(?:\.git|\.agents|\.gitnexus|\.claude|\.cache|artifacts|data|models|recordings|sessions|transcripts|exports|env|\.venv|\.runtimes|build|dist|checkpoints)(?:\/|$)/i;
const privateExtensions = /\.(?:wav|flac|mp3|m4a|ogg|opus|aac|aif|aiff|wma|mp4|webm|srt|vtt|rttm|stm|bin|safetensors|onnx|ckpt|pt|pth|gguf|ggml|pfx|p12|pem|key|jks|keystore|exe|dll|pyd|log|pyc)$/i;
const privateMetadata = /(?:[A-Za-z]:[\\/]Users[\\/](?!<)[^\\/\r\n]+[\\/]|\b\d{8}_\d{6}_\d{6}_(?:mic|system)\.wav\b|\bphase0_\d{6}_\d{4})/;

function git(root, args, input, allowedCodes = [0]) {
  const result = spawnSync('git', ['-c', 'core.excludesFile=', ...args], {
    cwd: root, input, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (!allowedCodes.includes(result.status)) throw new Error(result.stderr.trim() || 'Git command failed.');
  return result.stdout;
}

function safePath(root, file) {
  if (file.split('/').some(part => part === '.git' || part === '..')) {
    throw new Error(`Percorso non consentito: ${file}`);
  }
  const absolute = path.resolve(root, file);
  if (!absolute.startsWith(root + path.sep)) throw new Error(`Percorso esterno: ${file}`);
  let current = root;
  for (const part of path.relative(root, absolute).split(path.sep)) {
    current = path.join(current, part);
    try {
      if (fs.lstatSync(current).isSymbolicLink()) {
        throw new Error(`Link simbolico da verificare: ${file}`);
      }
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  }
  return absolute;
}

function ignored(root, files) {
  if (!files.length) return new Set();
  return new Set(git(root, ['check-ignore', '--no-index', '-z', '--stdin'],
    files.join('\0') + '\0', [0, 1]).split('\0').filter(Boolean));
}

function explicitlyPrivate(file) {
  const name = path.posix.basename(file);
  const template = ['.env.example', '.env.sample', '.env.template'].includes(name);
  return privateRoots.test(file) || privateExtensions.test(file)
    || name === '.hf_token' || (!template && /^\.env(?:\.|$)/.test(name));
}

function scanWithGitleaks(source, content) {
  const local = path.join(source, '.cache/publication-audit/gitleaks/gitleaks.exe');
  const command = fs.existsSync(local) ? local : 'gitleaks';
  const result = spawnSync(command, [
    'stdin', '--redact=100', '--no-banner', '--no-color', '--ignore-gitleaks-allow',
    '--max-decode-depth=2',
  ], { input: content, encoding: 'utf8', maxBuffer: 1024 * 1024 });
  if (result.error) throw new Error('Gitleaks non disponibile. Installalo nel PATH prima di usare --apply.');
  if (result.status !== 0) throw new Error('Scansione Gitleaks non superata. Nessun file sincronizzato.');
}

export function syncClean({ source, destination, apply = false, scan = scanWithGitleaks }) {
  if ([source, destination].some(root => fs.lstatSync(root).isSymbolicLink())) {
    throw new Error('Le cartelle dei repository non devono essere link simbolici.');
  }
  source = fs.realpathSync(source);
  destination = fs.realpathSync(destination);
  if (source === destination || path.dirname(source) !== path.dirname(destination)) {
    throw new Error('Lavoro e CLEAN devono essere due cartelle distinte nella stessa directory.');
  }
  for (const root of [source, destination]) {
    if (fs.realpathSync(git(root, ['rev-parse', '--show-toplevel']).trim()).toLowerCase() !== root.toLowerCase()) {
      throw new Error('Esegui il comando su due repository separati, senza link simbolici.');
    }
  }
  if (!fs.statSync(path.join(destination, '.git')).isDirectory()) {
    throw new Error('La CLEAN deve avere una propria cartella .git indipendente.');
  }
  const status = () => git(destination, ['status', '--porcelain=v1', '--untracked-files=all']);
  if (apply && status().trim()) throw new Error('La CLEAN contiene modifiche locali. Revisionale e committale prima di sincronizzare.');

  const candidates = [...new Set(git(source, ['ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    .split('\0').filter(Boolean))].sort();
  const excludedSource = ignored(source, candidates);
  const excludedClean = ignored(destination, candidates);
  const selected = candidates.filter(file => !excludedSource.has(file) && !excludedClean.has(file)
    && !explicitlyPrivate(file) && fs.existsSync(safePath(source, file)));
  const selectedSet = new Set(selected);
  const writes = [];
  const protectedDifferences = [];
  const payload = [];
  const tokenPath = path.join(source, '.hf_token');
  const token = fs.existsSync(tokenPath) ? fs.readFileSync(tokenPath, 'utf8').trim() : '';

  for (const file of selected) {
    const from = safePath(source, file);
    const to = safePath(destination, file);
    if (!fs.statSync(from).isFile()) throw new Error(`Elemento non supportato: ${file}`);
    let original = fs.readFileSync(from);
    const existing = fs.existsSync(to) ? fs.readFileSync(to) : null;
    if (protectedFiles.has(file)) {
      if (!existing) throw new Error(`File pubblico protetto mancante nella CLEAN: ${file}`);
      if (!original.equals(existing)) protectedDifferences.push(file);
      payload.push(Buffer.from(`\nFILE ${file}\n`), existing);
      continue;
    }
    if (file.endsWith('.md')) {
      const localPrefix = 'file:///' + source.replaceAll(path.sep, '/') + '/';
      const escaped = localPrefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const relative = path.posix.relative(path.posix.dirname(file), '.') || '.';
      original = Buffer.from(original.toString('utf8').replace(new RegExp(escaped, 'gi'), relative + '/'));
    }
    if (token.length >= 12 && original.includes(Buffer.from(token))) {
      throw new Error(`Token locale rilevato in ${file}. Nessun file sincronizzato.`);
    }
    if (!original.includes(0) && privateMetadata.test(original.toString('utf8'))) {
      throw new Error(`Metadati di una registrazione o percorso personale in ${file}. Revisione necessaria.`);
    }
    payload.push(Buffer.from(`\nFILE ${file}\n`), original);
    if (!existing || !original.equals(existing)) {
      writes.push({ file, content: original, type: existing ? 'UPDATE' : 'ADD' });
    }
  }
  const removals = git(destination, ['ls-files', '-z']).split('\0').filter(Boolean)
    .filter(file => !selectedSet.has(file) && !protectedFiles.has(file));
  for (const file of removals) safePath(destination, file);
  for (const required of ['LICENSE.md', 'NOTICE.md']) {
    if (!selectedSet.has(required)) throw new Error(`File di licenza mancante o escluso: ${required}`);
  }
  const content = Buffer.concat(payload);
  if (token.length >= 12 && content.includes(Buffer.from(token))) {
    throw new Error('Token locale rilevato nei file pubblici protetti. Nessun file sincronizzato.');
  }

  if (apply && (writes.length || removals.length)) {
    scan(source, content);
    if (status().trim()) throw new Error('La CLEAN e cambiata durante la verifica. Operazione interrotta.');
    for (const item of writes) {
      const target = safePath(destination, item.file);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, item.content);
    }
    for (const file of removals) fs.unlinkSync(safePath(destination, file));
  }
  return {
    applied: apply,
    changes: [...writes.map(({ file, type }) => ({ file, type })),
      ...removals.map(file => ({ file, type: 'DELETE' }))],
    protectedDifferences,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const args = process.argv.slice(2);
    if (args.some(arg => arg !== '--apply')) throw new Error('Uso: node scripts/sync-clean.mjs [--apply]');
    const source = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
    const destination = path.resolve(source, '..', 'tessitura-github');
    const result = syncClean({ source, destination, apply: args.includes('--apply') });
    console.log(`${result.applied ? 'Sincronizzazione applicata' : 'Anteprima'}: ${result.changes.length} modifiche.`);
    for (const change of result.changes) console.log(`${change.type} ${change.file}`);
    if (result.protectedDifferences.length) {
      console.log('File pubblici personalizzati da mantenere manualmente:');
      for (const file of result.protectedDifferences) console.log(`PROTECTED ${file}`);
    }
    console.log('Nessun git add, commit o push automatico.');
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
