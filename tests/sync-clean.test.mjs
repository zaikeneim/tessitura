import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { syncClean } from '../scripts/sync-clean.mjs';

function git(root, ...args) {
  const result = spawnSync('git', ['-c', 'core.excludesFile=', ...args], {
    cwd: root, encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout;
}

function write(root, file, content) {
  const target = path.join(root, file);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, content);
}

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'tessitura-sync-'));
  const source = path.join(root, 'work');
  const destination = path.join(root, 'clean');
  for (const directory of [source, destination]) {
    fs.mkdirSync(directory);
    git(directory, 'init', '--initial-branch=main');
    git(directory, 'config', 'user.name', 'Sync Test');
    git(directory, 'config', 'user.email', 'sync@example.invalid');
    git(directory, 'config', 'core.autocrlf', 'false');
    git(directory, 'config', 'commit.gpgSign', 'false');
    write(directory, '.gitignore', 'data/\nmodels/\n.hf_token\n.cache/\n');
    write(directory, 'LICENSE.md', 'Test license\n');
    write(directory, 'NOTICE.md', 'Test notice\n');
    write(directory, 'README.md', directory === source ? 'Local README\n' : 'Public README\n');
    write(directory, 'src/app.py', 'old\n');
    write(directory, 'src/removed.py', 'old\n');
    git(directory, 'add', '--all');
    git(directory, 'commit', '--quiet', '-m', 'test fixture');
  }
  fs.unlinkSync(path.join(source, 'src/removed.py'));
  write(source, 'src/app.py', 'updated\n');
  write(source, 'src/new.py', 'new\n');
  write(source, '.hf_token', 'private-local-value');
  write(source, 'data/session.txt', 'private transcript');
  write(destination, 'data/local.txt', 'keep local data');
  t.after(() => {
    const resolved = fs.realpathSync(root);
    const temporary = fs.realpathSync(os.tmpdir());
    assert.ok(resolved.startsWith(temporary + path.sep));
    fs.rmSync(resolved, { recursive: true, force: true });
  });
  return { source, destination };
}

test('preview preserves both repositories and filters private files', t => {
  const locations = fixture(t);
  const before = git(locations.source, 'status', '--porcelain');
  const result = syncClean(locations);
  assert.deepEqual(result.changes, [
    { file: 'src/app.py', type: 'UPDATE' },
    { file: 'src/new.py', type: 'ADD' },
    { file: 'src/removed.py', type: 'DELETE' },
  ]);
  assert.deepEqual(result.protectedDifferences, ['README.md']);
  assert.equal(fs.readFileSync(path.join(locations.destination, 'src/app.py'), 'utf8'), 'old\n');
  assert.equal(git(locations.destination, 'status', '--porcelain'), '');
  assert.equal(git(locations.source, 'status', '--porcelain'), before);
});

test('apply scans before writing and preserves public overrides and local data', t => {
  const locations = fixture(t);
  const sourceHead = git(locations.source, 'rev-parse', 'HEAD');
  const cleanHead = git(locations.destination, 'rev-parse', 'HEAD');
  let scanned = false;
  const result = syncClean({ ...locations, apply: true, scan: (_source, payload) => {
    assert.ok(!payload.includes(Buffer.from('private transcript')));
    assert.equal(fs.readFileSync(path.join(locations.destination, 'src/app.py'), 'utf8'), 'old\n');
    scanned = true;
  } });
  assert.ok(scanned);
  assert.equal(result.changes.length, 3);
  assert.equal(fs.readFileSync(path.join(locations.destination, 'src/app.py'), 'utf8'), 'updated\n');
  assert.equal(fs.readFileSync(path.join(locations.destination, 'README.md'), 'utf8'), 'Public README\n');
  assert.equal(fs.readFileSync(path.join(locations.destination, 'data/local.txt'), 'utf8'), 'keep local data');
  assert.equal(fs.existsSync(path.join(locations.destination, 'src/removed.py')), false);
  assert.equal(fs.existsSync(path.join(locations.destination, '.hf_token')), false);
  assert.equal(git(locations.source, 'rev-parse', 'HEAD'), sourceHead);
  assert.equal(git(locations.destination, 'rev-parse', 'HEAD'), cleanHead);
});

test('dirty destination and a failed scan block writes', t => {
  const locations = fixture(t);
  write(locations.destination, 'src/app.py', 'user edit\n');
  assert.throws(() => syncClean({ ...locations, apply: true }), /modifiche locali/);
  assert.equal(fs.readFileSync(path.join(locations.destination, 'src/app.py'), 'utf8'), 'user edit\n');
  git(locations.destination, 'restore', 'src/app.py');
  assert.throws(() => syncClean({ ...locations, apply: true, scan: () => { throw Error('scan failed'); } }), /scan failed/);
  assert.equal(git(locations.destination, 'status', '--porcelain'), '');
});

test('an already tracked private file and an exact local token cannot leak', t => {
  const locations = fixture(t);
  git(locations.source, 'add', '--force', '.hf_token');
  assert.ok(syncClean(locations).changes.every(change => change.file !== '.hf_token'));
  write(locations.source, 'src/leak.py', 'value = "private-local-value"\n');
  assert.throws(() => syncClean(locations), /Token locale rilevato/);
  assert.equal(git(locations.destination, 'status', '--porcelain'), '');
});

test('local documentation links become portable without changing the source', t => {
  const locations = fixture(t);
  const localLink = 'file:///' + locations.source.replaceAll(path.sep, '/') + '/src/app.py';
  write(locations.source, 'docs/guide.md', `[Source](${localLink})\n`);
  syncClean({ ...locations, apply: true, scan: () => {} });
  assert.equal(fs.readFileSync(path.join(locations.destination, 'docs/guide.md'), 'utf8'), '[Source](../src/app.py)\n');
  assert.equal(fs.readFileSync(path.join(locations.source, 'docs/guide.md'), 'utf8'), `[Source](${localLink})\n`);
});

test('public overrides are also checked for the exact local token', t => {
  const locations = fixture(t);
  write(locations.destination, 'README.md', 'private-local-value\n');
  git(locations.destination, 'add', 'README.md');
  git(locations.destination, 'commit', '--quiet', '-m', 'override fixture');
  assert.throws(() => syncClean(locations), /Token locale rilevato/);
});

test('destination directory links cannot redirect writes outside the CLEAN', t => {
  const locations = fixture(t);
  const outside = path.join(path.dirname(locations.destination), 'outside');
  fs.mkdirSync(outside);
  fs.symlinkSync(outside, path.join(locations.destination, 'src/linked'), process.platform === 'win32' ? 'junction' : 'dir');
  write(locations.source, 'src/linked/new.py', 'new\n');
  assert.throws(() => syncClean(locations), /Link simbolico/);
  assert.deepEqual(fs.readdirSync(outside), []);
});
