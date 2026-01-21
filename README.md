# Binary Search & Optimization - Refactored Package Collection

**Version 1.0.0** - Major refactor from combined v0.1 package

This repository contains two independent, production-ready Python packages for mathematical optimization and search algorithms.

## 🎯 What Changed in v1.0

### v0.1 (Old Structure) ❌
- Single combined package with unrelated classes  
- Critical bugs (missing imports, overflow errors)
- No tests, no documentation
- Poor maintainability

### v1.0 (New Structure) ✅
- **Two separate packages** for different use cases
- **All critical bugs fixed**
- **98% & 76% test coverage** (59 tests total)
- **Comprehensive documentation** with examples
- **Type hints and input validation** throughout
- **CI/CD with GitHub Actions**

## 📦 Packages

### 1. Binary Rate Optimizer
**Gradient descent with automatic learning rate tuning**

### 2. Binary Search Algorithms  
**Advanced search with tolerance-based comparisons**

See individual package READMEs for full documentation.

## 🚀 Quick Installation

```bash
# Binary Rate Optimizer
cd binary_rate_optimizer && pip install -e .

# Binary Search Algorithms
cd binary_search_algorithms && pip install -e .
```

## ✅ Testing & Coverage

```bash
pytest tests/ --cov=binary_rate_optimizer --cov=binary_search_algorithms
```

**Results**: 59 tests, 81% coverage, <1 second runtime

## 📚 Full Documentation

- [Binary Rate Optimizer README](binary_rate_optimizer/README.md)
- [Binary Search Algorithms README](binary_search_algorithms/README.md)
- [Refactoring Decisions](DECISIONS.md)

---

**Refactored following Simplicity Protocol 3 best practices**
