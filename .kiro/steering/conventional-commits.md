---
inclusion: always
---

# Conventional Commits

All commits in this repository MUST follow the Conventional Commits specification.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (white-space, formatting, etc)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

## Scope (optional)

The scope should be the name of the affected component or area (e.g., `ui`, `api`, `auth`, `config`).

## Subject

- Use imperative, present tense: "change" not "changed" nor "changes"
- Don't capitalize first letter
- No period (.) at the end
- Maximum 50 characters

## Body (optional)

- Use imperative, present tense
- Include motivation for the change and contrast with previous behavior
- Wrap at 72 characters

## Footer (optional)

- Reference issues: `Closes #123` or `Fixes #456`
- Breaking changes: Start with `BREAKING CHANGE:` followed by description

## Examples

```
feat(auth): add user login functionality

Implement JWT-based authentication with refresh tokens.
Users can now log in with email and password.

Closes #42
```

```
fix(ui): correct button alignment on mobile

The submit button was misaligned on screens smaller than 768px.
Updated flexbox properties to center the button properly.
```

```
docs: update installation instructions
```

```
refactor(api): simplify error handling logic
```

## Breaking Changes

If a commit introduces breaking changes, add `!` after the type/scope:

```
feat(api)!: change authentication endpoint

BREAKING CHANGE: The /auth endpoint now requires a different payload structure.
Old: { username, password }
New: { email, password }
```

## Gitmoji

Prefix commit subjects with a gitmoji to visually categorize changes. Use the most relevant emoji from this table:

| Emoji | Code                      | Use when                    |
| ----- | ------------------------- | --------------------------- |
| ✨    | `:sparkles:`              | New feature                 |
| 🐛    | `:bug:`                   | Bug fix                     |
| 📝    | `:memo:`                  | Documentation               |
| 🎨    | `:art:`                   | Code structure / formatting |
| ⚡️    | `:zap:`                   | Performance improvement     |
| 🔥    | `:fire:`                  | Removing code or files      |
| 🚑️    | `:ambulance:`             | Critical hotfix             |
| ♻️    | `:recycle:`               | Refactor                    |
| ✅    | `:white_check_mark:`      | Tests                       |
| 🔒️    | `:lock:`                  | Security fix                |
| 🚀    | `:rocket:`                | Deploy                      |
| 💥    | `:boom:`                  | Breaking change             |
| ➕    | `:heavy_plus_sign:`       | Add dependency              |
| ➖    | `:heavy_minus_sign:`      | Remove dependency           |
| ⬆️    | `:arrow_up:`              | Upgrade dependency          |
| ⬇️    | `:arrow_down:`            | Downgrade dependency        |
| 🔧    | `:wrench:`                | Configuration               |
| 🔨    | `:hammer:`                | Dev scripts                 |
| 👷    | `:construction_worker:`   | CI build system             |
| 💚    | `:green_heart:`           | Fix CI                      |
| 🚧    | `:construction:`          | Work in progress            |
| 📦️    | `:package:`               | Compiled files / packages   |
| 🚚    | `:truck:`                 | Move / rename files         |
| 📄    | `:page_facing_up:`        | License                     |
| 🙈    | `:see_no_evil:`           | Gitignore                   |
| 🏗️    | `:building_construction:` | Architectural change        |
| 🧱    | `:bricks:`                | Infrastructure              |
| ⚗️    | `:alembic:`               | Experiment                  |
| 🩹    | `:adhesive_bandage:`      | Simple non-critical fix     |
| ⚰️    | `:coffin:`                | Remove dead code            |
| 💡    | `:bulb:`                  | Source code comments        |
| ✏️    | `:pencil2:`               | Fix typo                    |
| ⏪️    | `:rewind:`                | Revert changes              |
| 🔖    | `:bookmark:`              | Release / version tag       |

### Format

The emoji goes after the type/scope prefix:

```
<type>(<scope>): <emoji> <subject>
```

### Examples

```
feat(model): ✨ add early stopping with patience parameter
```

```
fix(eval): 🐛 handle passthrough SI-SDR as absolute metric
```

```
docs: 📖 update README with setup instructions
```
