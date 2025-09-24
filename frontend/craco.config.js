const path = require('path');

const config = {
  disableHotReload: process.env.DISABLE_HOT_RELOAD === 'true',
};

module.exports = {
  babel: {
    presets: ["@babel/preset-react"],
  },
  eslint: {
    enable: false, // ✅ This disables ESLint entirely
  },
  webpack: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
    configure: (webpackConfig) => {
      if (config.disableHotReload) {
        webpackConfig.plugins = webpackConfig.plugins.filter(
          plugin => plugin.constructor.name !== 'HotModuleReplacementPlugin'
        );
        webpackConfig.watch = false;
        webpackConfig.watchOptions = {
          ignored: /.*/,
        };
      } else {
        webpackConfig.watchOptions = {
          ...webpackConfig.watchOptions,
          ignored: [
            '**/node_modules/**',
            '**/.git/**',
            '**/build/**',
            '**/dist/**',
            '**/coverage/**',
            '**/public/**',
          ],
        };
      }

      return webpackConfig;
    },
  },
};
