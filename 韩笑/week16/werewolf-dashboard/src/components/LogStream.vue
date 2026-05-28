<template>
  <div class="log-stream">
    <div class="log-header">
      <h3>事件日志流</h3>
      <div class="log-filters">
        <el-radio-group v-model="store.logFilter" size="small">
          <el-radio-button value="all">全部</el-radio-button>
          <el-radio-button value="system">系统</el-radio-button>
          <el-radio-button value="speech">发言</el-radio-button>
          <el-radio-button value="action">行动</el-radio-button>
          <el-radio-button value="private">私密</el-radio-button>
        </el-radio-group>
        <el-button size="small" @click="autoScroll = !autoScroll" :type="autoScroll ? 'primary' : ''">
          {{ autoScroll ? '自动滚动' : '暂停滚动' }}
        </el-button>
        <el-button size="small" @click="store.logs = []">清空</el-button>
      </div>
    </div>

    <div class="log-container" ref="logContainer">
      <div
        v-for="log in store.displayedLogs"
        :key="log.id"
        class="log-entry"
        :class="'log-' + log.type"
      >
        <span class="log-time">{{ formatTime(log.timestamp) }}</span>
        <span class="log-round">[R{{ log.round }}]</span>
        <el-tag :type="logTagType(log.type)" size="small" effect="dark">
          {{ logTypeLabel(log.type) }}
        </el-tag>
        <span class="log-content">
          <!-- 系统消息 -->
          <template v-if="log.type === 'system'">
            {{ log.content.message }}
          </template>
          <!-- 发言 -->
          <template v-else-if="log.type === 'speech'">
            <strong>{{ log.content.playerId }}号{{ log.content.playerName }}:</strong>
            <span class="speech-text">"{{ log.content.content?.slice(0, 300) }}"</span>
            <el-button
              v-if="log.content.thought"
              link
              type="primary"
              size="small"
              @click="showThought(log.content)"
            >
              查看内心独白
            </el-button>
          </template>
          <!-- 行动 -->
          <template v-else-if="log.type === 'action'">
            <span v-if="log.content.phase">阶段: {{ log.content.phase }}</span>
            <span v-if="log.content.validActions">
              可选行动: {{ log.content.validActions?.join(', ') }}
            </span>
            <span v-if="log.content.tally">
              投票结果: {{ JSON.stringify(log.content.tally) }}
            </span>
          </template>
          <!-- 私密 -->
          <template v-else-if="log.type === 'private'">
            [私密] {{ log.content.infoType }}: {{ JSON.stringify(log.content.payload).slice(0, 200) }}
          </template>
        </span>
      </div>

      <el-empty v-if="store.displayedLogs.length === 0" description="暂无事件" :image-size="60" />
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { useGameStore } from '../stores/game'

const store = useGameStore()
const logContainer = ref(null)
const autoScroll = ref(true)

function formatTime(ts) {
  const d = new Date(ts)
  return d.toLocaleTimeString('zh-CN', { hour12: false })
}

function logTagType(type) {
  const map = { system: 'info', speech: 'success', action: 'warning', private: 'danger' }
  return map[type] || 'info'
}

function logTypeLabel(type) {
  const map = { system: '系统', speech: '发言', action: '行动', private: '私密' }
  return map[type] || type
}

function showThought(content) {
  if (content.thought) {
    ElMessageBox.alert(content.thought, `${content.playerId}号的内心独白`, {
      confirmButtonText: '关闭',
      customClass: 'thought-dialog',
    })
  }
}

// 自动滚动
watch(() => store.displayedLogs.length, () => {
  if (autoScroll.value) {
    nextTick(() => {
      if (logContainer.value) {
        logContainer.value.scrollTop = logContainer.value.scrollHeight
      }
    })
  }
})
</script>

<style scoped>
.log-stream {
  background: #1e1e2e; border-radius: 8px; padding: 16px;
  display: flex; flex-direction: column; height: 100%;
}
.log-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px; flex-wrap: wrap; gap: 8px;
}
.log-header h3 { margin: 0; color: #ecf0f1; font-size: 16px; }
.log-filters { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.log-container {
  flex: 1; overflow-y: auto; min-height: 200px; max-height: 400px;
  background: #16162a; border-radius: 6px; padding: 8px;
}
.log-entry {
  display: flex; gap: 8px; align-items: flex-start; padding: 6px 8px;
  border-bottom: 1px solid #2d2d44; font-size: 13px; color: #ecf0f1;
}
.log-entry:hover { background: #2d2d44; }
.log-system { border-left: 2px solid #3498db; }
.log-speech { border-left: 2px solid #2ecc71; }
.log-action { border-left: 2px solid #f39c12; }
.log-private { border-left: 2px solid #e74c3c; }
.log-time { color: #7f8c8d; font-size: 11px; white-space: nowrap; min-width: 70px; }
.log-round { color: #95a5a6; font-size: 11px; white-space: nowrap; }
.log-content { flex: 1; word-break: break-all; }
.speech-text { color: #2ecc71; font-style: italic; }
</style>
